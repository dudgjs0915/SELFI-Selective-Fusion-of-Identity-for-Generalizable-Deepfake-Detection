# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Two-phase training code with variability-based continual learning

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import json
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import wandb

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter


parser = argparse.ArgumentParser(description='Two-phase training with variability-based sample selection.')
parser.add_argument('--detector_path', type=str,
                    default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
parser.add_argument('--wandb', action='store_true', default=False, help='enable WandB logging')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode: 3 epochs, 1/100 train data, 1/10 test data')
parser.add_argument('--dataset_json_folder', type=str, default=None, help='path to dataset json folder')
parser.add_argument('--skip_phase0', action='store_true', default=False, help='skip Phase 0 and start from Phase 1 (for debugging)')
parser.add_argument('--phase0_log_dir', type=str, default=None, help='path to existing Phase 0 log directory (required if --skip_phase0 is used)')
parser.add_argument('--use_pretrained_for_phase1', action='store_true', default=False, help='use pretrained weights instead of Phase 0 checkpoint for Phase 1 (to measure variability-based filtering effect without Phase 0 training)')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config, allowed_frames=None):
    """
    Prepare training data loader with optional frame filtering for Phase 1.
    
    Args:
        config: configuration dictionary
        allowed_frames: Optional set of frame paths to include (for Phase 1 filtering)
    """
    # Store allowed_frames in config for dataset to access
    if allowed_frames is not None:
        config['allowed_frames'] = allowed_frames
    else:
        config['allowed_frames'] = None
    
    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='train',
                )
    
    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler, 
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                )
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
        else:
            test_set = LRLDataset(
                config=config,
                mode='test',
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last = (test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config, phase='phase0'):
    """
    Choose optimizer with phase-specific learning rate.
    
    Args:
        model: the neural network model
        config: configuration dictionary
        phase: 'phase0' or 'phase1'
    """
    opt_name = config['optimizer']['type']
    
    # Get learning rate based on phase
    if phase == 'phase1' and config.get('phase1_enabled', False):
        lr = config.get('phase1_lr', config['optimizer'][opt_name]['lr'])
    else:
        lr = config['optimizer'][opt_name]['lr']
    
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=lr,
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer, phase='phase0'):
    """
    Choose scheduler with phase-specific parameters.
    
    Args:
        config: configuration dictionary
        optimizer: the optimizer
        phase: 'phase0' or 'phase1'
    """
    if config['lr_scheduler'] is None:
        return None
    
    # Get number of epochs based on phase
    if phase == 'phase1' and config.get('phase1_enabled', False):
        nEpochs = config.get('phase1_nEpochs', config['nEpochs'])
    else:
        nEpochs = config['nEpochs']
    
    if config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('lr_T_max', nEpochs),
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            nEpochs,
            int(nEpochs/4),
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def filter_high_variability_samples(training_dynamics_path, top_percent, config, logger):
    """
    Filter samples based on variability from training dynamics.
    Selects top N% samples separately for real (label=0) and fake (label=1).
    
    Args:
        training_dynamics_path: path to training_dynamics.json
        top_percent: percentage of top variability samples to keep (e.g., 20)
        config: configuration dictionary
        logger: logger instance
        
    Returns:
        set of frame paths to keep
    """
    logger.info(f"Loading training dynamics from: {training_dynamics_path}")
    
    if not os.path.exists(training_dynamics_path):
        logger.error(f"Training dynamics file not found: {training_dynamics_path}")
        raise FileNotFoundError(f"Training dynamics file not found: {training_dynamics_path}")
    
    with open(training_dynamics_path, 'r') as f:
        dynamics_data = json.load(f)
    
    if len(dynamics_data) == 0:
        logger.error("Training dynamics file is empty!")
        raise ValueError("Training dynamics file is empty!")
    
    logger.info(f"Loaded {len(dynamics_data)} samples from training dynamics")
    
    # Separate samples by label
    real_samples = [d for d in dynamics_data if d['label'] == 0]
    fake_samples = [d for d in dynamics_data if d['label'] == 1]
    
    logger.info(f"Real samples: {len(real_samples)}, Fake samples: {len(fake_samples)}")
    
    # Sort by variability (descending)
    real_samples_sorted = sorted(real_samples, key=lambda x: x['variability'], reverse=True)
    fake_samples_sorted = sorted(fake_samples, key=lambda x: x['variability'], reverse=True)
    
    # Calculate number of samples to keep
    num_real_to_keep = max(1, int(len(real_samples) * top_percent / 100))
    num_fake_to_keep = max(1, int(len(fake_samples) * top_percent / 100))
    
    # Select top N%
    selected_real = real_samples_sorted[:num_real_to_keep]
    selected_fake = fake_samples_sorted[:num_fake_to_keep]
    
    logger.info(f"Selected top {top_percent}% variability samples:")
    logger.info(f"  Real: {num_real_to_keep}/{len(real_samples)} ({num_real_to_keep/len(real_samples)*100:.1f}%)")
    logger.info(f"  Fake: {num_fake_to_keep}/{len(fake_samples)} ({num_fake_to_keep/len(fake_samples)*100:.1f}%)")
    
    # Log variability statistics
    if len(selected_real) > 0:
        real_var_range = f"[{selected_real[-1]['variability']:.4f}, {selected_real[0]['variability']:.4f}]"
        logger.info(f"  Real variability range: {real_var_range}")
    if len(selected_fake) > 0:
        fake_var_range = f"[{selected_fake[-1]['variability']:.4f}, {selected_fake[0]['variability']:.4f}]"
        logger.info(f"  Fake variability range: {fake_var_range}")
    
    # Create set of frame paths
    allowed_frames = set()
    for sample in selected_real + selected_fake:
        allowed_frames.add(sample['path'])
    
    logger.info(f"Total filtered frames: {len(allowed_frames)}")
    
    # Validation check
    if len(allowed_frames) < 10:
        logger.warning(f"Very few samples selected ({len(allowed_frames)}). Consider increasing top_percent.")
    
    return allowed_frames


def run_training_phase(config, phase_name, phase_num_epochs, train_data_loader, test_data_loaders, 
                       model, optimizer, scheduler, logger, metric_scoring, timenow, checkpoint_to_load=None):
    """
    Run a single training phase.
    
    Args:
        config: configuration dictionary
        phase_name: 'phase0' or 'phase1'
        phase_num_epochs: number of epochs for this phase
        train_data_loader: training data loader
        test_data_loaders: dictionary of test data loaders
        model: the neural network model
        optimizer: optimizer instance (will be recreated for Phase 1 after LoRA activation)
        scheduler: scheduler instance (will be recreated for Phase 1 after LoRA activation)
        logger: logger instance
        metric_scoring: metric for evaluation
        timenow: timestamp string
        checkpoint_to_load: path to checkpoint to load (for Phase 1)
        
    Returns:
        best_metric: best metric achieved in this phase
        phase_log_dir: log directory for this phase
    """
    # Load checkpoint if provided (for Phase 1 continual learning)
    if checkpoint_to_load is not None and os.path.exists(checkpoint_to_load):
        logger.info(f"Loading checkpoint from Phase 0: {checkpoint_to_load}")
        checkpoint = torch.load(checkpoint_to_load, map_location='cpu')
        if 'state_dict' in checkpoint:
            # SVDD model format
            model.load_state_dict(checkpoint['state_dict'])
            if hasattr(model, 'R'):
                model.R = checkpoint['R']
            if hasattr(model, 'c'):
                model.c = checkpoint['c']
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Successfully loaded Phase 0 checkpoint for continual learning")
        
        # Enable LoRA for Phase 1 if this is Phase 1 and model supports it
        if phase_name == 'phase1' and hasattr(model, 'enable_lora'):
            logger.info("Enabling LoRA for Phase 1...")
            lora_config = config.get('lora_config', {})
            lora_r = lora_config.get('r', 8)
            lora_alpha = lora_config.get('alpha', 16)
            lora_dropout = lora_config.get('dropout', 0.1)
            lora_target_modules = lora_config.get('target_modules', None)
            
            model.enable_lora(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules
            )
            model.freeze_non_lora()
            logger.info("LoRA enabled and non-LoRA parameters frozen for Phase 1")
            
            # CRITICAL: Recreate optimizer and scheduler AFTER LoRA activation
            # The optimizer needs to know about the newly added LoRA parameters
            logger.info("Recreating optimizer and scheduler with LoRA parameters...")
            optimizer = choose_optimizer(model, config, phase='phase1')
            scheduler = choose_scheduler(config, optimizer, phase='phase1')
            logger.info("Optimizer and scheduler recreated with LoRA parameters")
            
    elif checkpoint_to_load is not None:
        logger.error(f"Checkpoint not found: {checkpoint_to_load}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_to_load}")
    
    # Prepare the trainer with phase information
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow, phase=phase_name)
    
    # Training loop
    logger.info(f"{'='*20} Starting {phase_name.upper()} {'='*20}")
    logger.info(f"Training for {phase_num_epochs} epochs")
    
    best_metric = None
    for epoch in range(1, phase_num_epochs + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
        
        # Update scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
    
    logger.info(f"{phase_name.upper()} completed with best testing metric: {parse_metric_for_print(best_metric)}")
    
    # Save training dynamics for this phase
    trainer.save_training_dynamics()
    
    # Save test dynamics for each test dataset
    for dataset_name in test_data_loaders.keys():
        trainer.save_test_dynamics(dataset_name)
    
    # Find best checkpoint for this phase (in test/ subdirectory)
    checkpoint_path = None
    
    # First check for 'avg' checkpoint (saved when multiple test datasets)
    avg_ckpt = os.path.join(trainer.log_dir, 'test', 'avg', 'ckpt_best.pth')
    if os.path.exists(avg_ckpt):
        checkpoint_path = avg_ckpt
        logger.info(f"{phase_name.upper()} best checkpoint (avg): {checkpoint_path}")
    else:
        # Then check individual test datasets
        for test_dataset_name in config['test_dataset']:
            potential_best_ckpt = os.path.join(trainer.log_dir, 'test', test_dataset_name, 'ckpt_best.pth')
            if os.path.exists(potential_best_ckpt):
                checkpoint_path = potential_best_ckpt
                logger.info(f"{phase_name.upper()} best checkpoint ({test_dataset_name}): {checkpoint_path}")
                break
    
    # If no best checkpoint exists, raise error
    if checkpoint_path is None:
        error_msg = f"No best checkpoint found for {phase_name}. Make sure save_ckpt=true in config and test is performed during training."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Return last step for next phase to use as offset
    last_step = trainer.last_step
    return best_metric, trainer.log_dir, checkpoint_path, trainer, last_step


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    if args.dataset_json_folder:
        config['dataset_json_folder'] = args.dataset_json_folder
    elif config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    
    # Check if two-phase training is enabled
    if not config.get('phase1_enabled', False):
        raise ValueError("phase1_enabled is not set to True in config. Use regular train.py for single-phase training.")
    
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Determine logger path
    # If resuming from Phase 0, use the existing log directory
    if args.skip_phase0 and args.phase0_log_dir:
        logger_path = args.phase0_log_dir
        logger = create_logger(os.path.join(logger_path, 'training.log'))  # Will append to existing log
        logger.info('='*80)
        logger.info(f'Resuming two-phase training from existing Phase 0')
        logger.info(f'Using existing log directory: {logger_path}')
        logger.info(f'Resume time: {timenow}')
        logger.info('='*80)
    else:
        # Create new log directory for fresh two-phase training
        task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
        logger_path = os.path.join(
            config['log_dir'],
            config['model_name'] + task_str + '_twophase_' + timenow
        )
        os.makedirs(logger_path, exist_ok=True)
        logger = create_logger(os.path.join(logger_path, 'training.log'))
        logger.info('Save log to {}'.format(logger_path))
    
    # Initialize WandB if --wandb flag is set
    wandb_run = None
    if args.wandb:
        try:
            import secrete_project_config
            wandb_run = wandb.init(
                project=secrete_project_config.project,
                entity=secrete_project_config.entity,
                name=f"{config['model_name']}_twophase_{timenow}",
                config=config,
                tags=[config['model_name'], timenow, "two_phase"]
            )
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            wandb_run = None
    config['wandb_run'] = wandb_run
    
    # Set base_log_dir for Trainer to use the same root directory
    config['base_log_dir'] = logger_path
    config['ddp']= args.ddp
    
    # Debug mode settings
    if args.debug:
        config['debug'] = True
        config['nEpochs'] = 3
        config['phase1_nEpochs'] = min(2, config.get('phase1_nEpochs', 2))
        config['train_sample_ratio'] = 0.01  # 1/100
        config['test_sample_ratio'] = 0.1    # 1/10
        logger.info("Debug mode enabled: Phase0=3 epochs, Phase1=2 epochs, 1/100 train data, 1/10 test data")
    else:
        config['debug'] = False
        config['train_sample_ratio'] = 1.0
        config['test_sample_ratio'] = 1.0
    
    # print configuration
    logger.info("--------------- Two-Phase Training Configuration ---------------")
    logger.info(f"Phase 0 epochs: {config['nEpochs']}")
    logger.info(f"Phase 1 enabled: {config.get('phase1_enabled', False)}")
    logger.info(f"Phase 1 epochs: {config.get('phase1_nEpochs', 'N/A')}")
    logger.info(f"Phase 1 LR: {config.get('phase1_lr', 'N/A')}")
    logger.info(f"Phase 1 variability top %: {config.get('phase1_variability_top_percent', 'N/A')}")
    if config.get('lora_config'):
        lora_config = config.get('lora_config', {})
        logger.info(f"LoRA enabled for Phase 1: r={lora_config.get('r', 8)}, alpha={lora_config.get('alpha', 16)}, dropout={lora_config.get('dropout', 0.1)}")
    logger.info(f"Skip Phase 0: {args.skip_phase0}")
    if args.skip_phase0 and args.phase0_log_dir:
        logger.info(f"Phase 0 Log Directory: {args.phase0_log_dir}")
    logger.info(f"Use Pretrained for Phase 1: {args.use_pretrained_for_phase1}")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
    
    # Prepare test data loaders (same for both phases)
    test_data_loaders = prepare_testing_data(config)
    
    # Prepare metric
    metric_scoring = choose_metric(config)
    
    phase0_checkpoint_path = None
    phase0_log_dir = None
    
    # ==================== PHASE 0: Full Dataset Training ====================
    if not args.skip_phase0:
        logger.info("\n" + "="*80)
        logger.info("PHASE 0: Training on full dataset")
        logger.info("="*80 + "\n")
        
        # Prepare Phase 0 data (no filtering)
        train_data_loader_phase0 = prepare_training_data(config, allowed_frames=None)
        
        # Prepare Phase 0 model
        model_class = DETECTOR[config['model_name']]
        model_phase0 = model_class(config)
        
        # Prepare Phase 0 optimizer and scheduler
        optimizer_phase0 = choose_optimizer(model_phase0, config, phase='phase0')
        scheduler_phase0 = choose_scheduler(config, optimizer_phase0, phase='phase0')
        
        # Run Phase 0 training
        best_metric_phase0, phase0_log_dir, phase0_checkpoint_path, trainer_phase0, phase0_last_step = run_training_phase(
            config=config,
            phase_name='phase0',
            phase_num_epochs=config['nEpochs'],
            train_data_loader=train_data_loader_phase0,
            test_data_loaders=test_data_loaders,
            model=model_phase0,
            optimizer=optimizer_phase0,
            scheduler=scheduler_phase0,
            logger=logger,
            metric_scoring=metric_scoring,
            timenow=timenow,
            checkpoint_to_load=None
        )
        
        logger.info(f"Phase 0 checkpoint ready at: {phase0_checkpoint_path}")
        logger.info(f"Phase 0 last step: {phase0_last_step}")
    else:
        logger.info("Skipping Phase 0 (--skip_phase0 flag set)")
        
        # Check if phase0_log_dir is provided
        if not args.phase0_log_dir:
            logger.error("--phase0_log_dir must be provided when using --skip_phase0")
            raise ValueError("--phase0_log_dir argument is required when skipping Phase 0")
        
        if not os.path.exists(args.phase0_log_dir):
            logger.error(f"Phase 0 log directory does not exist: {args.phase0_log_dir}")
            raise ValueError(f"Phase 0 log directory not found: {args.phase0_log_dir}")
        
        # phase0_log_dir should point to the base log directory
        # We'll construct the phase0 subdirectory path
        base_log_dir = args.phase0_log_dir
        phase0_log_dir = os.path.join(base_log_dir, 'phase0')
        
        if not os.path.exists(phase0_log_dir):
            logger.error(f"Phase 0 subdirectory does not exist: {phase0_log_dir}")
            raise ValueError(f"Phase 0 subdirectory not found. Expected: {phase0_log_dir}")
        
        logger.info(f"Using existing base log directory: {base_log_dir}")
        logger.info(f"Phase 0 directory: {phase0_log_dir}")
        
        # Find the best checkpoint from Phase 0
        # First check for 'avg' checkpoint
        avg_ckpt = os.path.join(phase0_log_dir, 'test', 'avg', 'ckpt_best.pth')
        if os.path.exists(avg_ckpt):
            phase0_checkpoint_path = avg_ckpt
            logger.info(f"Found Phase 0 checkpoint (avg): {phase0_checkpoint_path}")
        else:
            # Then check individual test datasets
            for test_dataset_name in config['test_dataset']:
                potential_best_ckpt = os.path.join(phase0_log_dir, 'test', test_dataset_name, 'ckpt_best.pth')
                if os.path.exists(potential_best_ckpt):
                    phase0_checkpoint_path = potential_best_ckpt
                    logger.info(f"Found Phase 0 checkpoint ({test_dataset_name}): {phase0_checkpoint_path}")
                    break
        
        if phase0_checkpoint_path is None:
            logger.error(f"No Phase 0 checkpoint found in {phase0_log_dir}")
            raise FileNotFoundError(f"No Phase 0 checkpoint found in {phase0_log_dir}")
        
        # Estimate Phase 0's last step for global_step_offset
        # We need to create a temporary data loader to get the size
        logger.info("Estimating Phase 0's last step...")
        temp_train_loader = prepare_training_data(config, allowed_frames=None)
        phase0_last_step = config['nEpochs'] * len(temp_train_loader)
        logger.info(f"Estimated Phase 0 last step: {phase0_last_step}")
        del temp_train_loader  # Clean up
    
    # ==================== Filter Samples Based on Variability ====================
    logger.info("\n" + "="*80)
    logger.info("FILTERING: Selecting high-variability samples")
    logger.info("="*80 + "\n")
    
    # Load training dynamics from Phase 0
    if phase0_log_dir is None:
        logger.error("Phase 0 log directory not available. Cannot filter samples.")
        raise ValueError("Phase 0 log directory required for filtering")
    
    training_dynamics_path = os.path.join(phase0_log_dir, 'training_dynamics.json')
    top_percent = config.get('phase1_variability_top_percent', 20)
    
    allowed_frames = filter_high_variability_samples(
        training_dynamics_path=training_dynamics_path,
        top_percent=top_percent,
        config=config,
        logger=logger
    )
    
    # ==================== PHASE 1: Continual Learning on High-Variability Samples ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: Continual learning on high-variability samples")
    logger.info("="*80 + "\n")
    
    # Prepare Phase 1 data (with filtering)
    train_data_loader_phase1 = prepare_training_data(config, allowed_frames=allowed_frames)
    
    # Prepare Phase 1 model (will load Phase 0 checkpoint inside run_training_phase)
    model_class = DETECTOR[config['model_name']]
    model_phase1 = model_class(config)
    
    # Prepare Phase 1 optimizer and scheduler (possibly with different hyperparameters)
    optimizer_phase1 = choose_optimizer(model_phase1, config, phase='phase1')
    scheduler_phase1 = choose_scheduler(config, optimizer_phase1, phase='phase1')
    
    phase1_nEpochs = config.get('phase1_nEpochs', config['nEpochs'])
    
    # Determine checkpoint to load for Phase 1
    if args.use_pretrained_for_phase1:
        # Use pretrained weights instead of Phase 0 checkpoint
        checkpoint_to_load_phase1 = None
        logger.info("Phase 1 will use pretrained weights (not Phase 0 checkpoint)")
        logger.info("This allows measuring the effect of variability-based filtering alone")
        # Don't set global_step_offset when using pretrained (start from 0)
        config['global_step_offset'] = 0
    else:
        # Use Phase 0 checkpoint (default behavior)
        checkpoint_to_load_phase1 = phase0_checkpoint_path
        logger.info(f"Phase 1 will load Phase 0 Best checkpoint: {phase0_checkpoint_path}")
        # Set global step offset for Phase 1 to continue from Phase 0
        config['global_step_offset'] = phase0_last_step
        logger.info(f"Phase 1 will start from global step {phase0_last_step}")
    
    # Run Phase 1 training
    best_metric_phase1, phase1_log_dir, phase1_checkpoint_path, trainer_phase1, phase1_last_step = run_training_phase(
        config=config,
        phase_name='phase1',
        phase_num_epochs=phase1_nEpochs,
        train_data_loader=train_data_loader_phase1,
        test_data_loaders=test_data_loaders,
        model=model_phase1,
        optimizer=optimizer_phase1,
        scheduler=scheduler_phase1,
        logger=logger,
        metric_scoring=metric_scoring,
        timenow=timenow,
        checkpoint_to_load=checkpoint_to_load_phase1
    )
    
    logger.info(f"Phase 1 checkpoint ready at: {phase1_checkpoint_path}")
    
    # ==================== Training Complete ====================
    logger.info("\n" + "="*80)
    logger.info("TWO-PHASE TRAINING COMPLETE")
    logger.info("="*80)
    if not args.skip_phase0:
        logger.info(f"Phase 0 best metric: {parse_metric_for_print(best_metric_phase0)}")
        logger.info(f"Phase 0 checkpoint: {phase0_checkpoint_path}")
    logger.info(f"Phase 1 best metric: {parse_metric_for_print(best_metric_phase1)}")
    logger.info(f"Phase 1 checkpoint: {phase1_checkpoint_path}")
    if hasattr(model_phase1, 'lora_enabled') and model_phase1.lora_enabled:
        logger.info("Phase 1 used LoRA fine-tuning for parameter-efficient training")
    logger.info(f"Logs saved to: {logger_path}")
    logger.info("="*80 + "\n")
    
    # Output log path for bash script parsing (DO NOT MODIFY THIS LINE FORMAT)
    print(f"TWOPHASE_LOG_DIR={logger_path}")
    
    # Stop WandB run if it was initialized
    if config.get('wandb_run') is not None:
        try:
            wandb.finish()
            logger.info("WandB run finished successfully")
        except Exception as e:
            logger.error(f"Error finishing WandB: {e}")


if __name__ == '__main__':
    main()
