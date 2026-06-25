# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None,
        phase='single'
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.wandb_run = config.get('wandb_run', None)  # Get WandB run from config
        self.phase = phase  # Training phase: 'single', 'phase0', or 'phase1'
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        # Training dynamics for Dataset Cartography
        self.training_dynamics = {}  # {sample_path: {'label': int, 'preds': [], 'prob_0': [], 'prob_1': []}}
        # Test dynamics for each test dataset
        self.test_dynamics = {}  # {dataset_name: {sample_path: {'label': int, 'preds': [], 'prob_0': [], 'prob_1': []}}}
        self.current_epoch = 0
        # Global step offset for Phase 1 to continue from Phase 0's last step
        self.global_step_offset = config.get('global_step_offset', 0)
        self.last_step = 0  # Track the last step for passing to next phase
        self.speed_up()  # move model to GPU

        # get current time
        self.timenow = time_now
        # create directory path
        # Use base_log_dir from config if provided (for two-phase training)
        if 'base_log_dir' in config:
            base_log_dir = config['base_log_dir']
        elif 'task_target' not in config:
            base_log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            base_log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        
        # Add phase subdirectory if not 'single' phase
        if self.phase != 'single':
            self.log_dir = os.path.join(base_log_dir, self.phase)
        else:
            self.log_dir = base_log_dir
        
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']],find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def save_training_dynamics(self):
        """
        Calculate and save training dynamics (Dataset Cartography metrics)
        Computes variability, confidence, and correctness for each sample
        """
        self.logger.info("Computing training dynamics statistics...")
        
        cartography_data = []
        
        for sample_path, dynamics in self.training_dynamics.items():
            # Filter out None values (missed epochs)
            valid_probs_0 = [p for p in dynamics['prob_0'] if p is not None]
            valid_probs_1 = [p for p in dynamics['prob_1'] if p is not None]
            valid_preds = [p for p in dynamics['preds'] if p is not None]
            
            if len(valid_probs_1) == 0:
                continue
            
            label = dynamics['label']
            
            # Calculate variability: standard deviation of probabilities
            # Use the probability of the true class for variability calculation
            if label == 0:
                probs_for_variability = valid_probs_0
            else:
                probs_for_variability = valid_probs_1
            variability = float(np.std(probs_for_variability))
            
            # Calculate confidence: mean probability of true class
            confidence = float(np.mean(probs_for_variability))
            
            # Calculate correctness: fraction of epochs where prediction was correct
            correct_predictions = [1 if pred == label else 0 for pred in valid_preds]
            correctness = float(np.mean(correct_predictions)) if len(correct_predictions) > 0 else 0.0
            
            cartography_data.append({
                'path': sample_path,
                'label': label,
                'preds': dynamics['preds'],
                'prob_0': dynamics['prob_0'],
                'prob_1': dynamics['prob_1'],
                'variability': variability,
                'confidence': confidence,
                'correctness': correctness
            })
        
        # Save to JSON file
        save_path = os.path.join(self.log_dir, 'training_dynamics.json')
        with open(save_path, 'w') as f:
            json.dump(cartography_data, f, indent=2)
        
        self.logger.info(f"Training dynamics saved to {save_path}")
        self.logger.info(f"Total samples tracked: {len(cartography_data)}")
        
        # Log statistics to WandB if available
        if self.wandb_run is not None:
            import wandb
            avg_variability = np.mean([d['variability'] for d in cartography_data])
            avg_confidence = np.mean([d['confidence'] for d in cartography_data])
            avg_correctness = np.mean([d['correctness'] for d in cartography_data])
            
            wandb.log({
                f"{self.phase}/cartography/avg_variability": avg_variability,
                f"{self.phase}/cartography/avg_confidence": avg_confidence,
                f"{self.phase}/cartography/avg_correctness": avg_correctness,
                f"{self.phase}/cartography/num_samples": len(cartography_data)
            })

    def save_test_dynamics(self, dataset_name):
        """
        Calculate and save test dynamics (Dataset Cartography metrics) for a specific test dataset
        Computes variability, confidence, and correctness for each sample
        """
        if dataset_name not in self.test_dynamics:
            self.logger.warning(f"No test dynamics found for dataset: {dataset_name}")
            return
        
        self.logger.info(f"Computing test dynamics statistics for {dataset_name}...")
        
        cartography_data = []
        dynamics_dict = self.test_dynamics[dataset_name]
        
        for sample_path, dynamics in dynamics_dict.items():
            # Filter out None values (missed epochs)
            valid_probs_0 = [p for p in dynamics['prob_0'] if p is not None]
            valid_probs_1 = [p for p in dynamics['prob_1'] if p is not None]
            valid_preds = [p for p in dynamics['preds'] if p is not None]
            
            if len(valid_probs_1) == 0:
                continue
            
            label = dynamics['label']
            
            # Calculate variability: standard deviation of probabilities
            # Use the probability of the true class for variability calculation
            if label == 0:
                probs_for_variability = valid_probs_0
            else:
                probs_for_variability = valid_probs_1
            variability = float(np.std(probs_for_variability))
            
            # Calculate confidence: mean probability of true class
            confidence = float(np.mean(probs_for_variability))
            
            # Calculate correctness: fraction of epochs where prediction was correct
            correct_predictions = [1 if pred == label else 0 for pred in valid_preds]
            correctness = float(np.mean(correct_predictions)) if len(correct_predictions) > 0 else 0.0
            
            cartography_data.append({
                'path': sample_path,
                'label': label,
                'preds': dynamics['preds'],
                'prob_0': dynamics['prob_0'],
                'prob_1': dynamics['prob_1'],
                'variability': variability,
                'confidence': confidence,
                'correctness': correctness
            })
        
        # Save to JSON file
        save_path = os.path.join(self.log_dir, f'test_dynamics_{dataset_name}.json')
        with open(save_path, 'w') as f:
            json.dump(cartography_data, f, indent=2)
        
        self.logger.info(f"Test dynamics for {dataset_name} saved to {save_path}")
        self.logger.info(f"Total samples tracked: {len(cartography_data)}")
        
        # Log statistics to WandB if available
        if self.wandb_run is not None:
            import wandb
            if len(cartography_data) > 0:
                avg_variability = np.mean([d['variability'] for d in cartography_data])
                avg_confidence = np.mean([d['confidence'] for d in cartography_data])
                avg_correctness = np.mean([d['correctness'] for d in cartography_data])
                
                wandb.log({
                    f"{self.phase}/test_cartography/{dataset_name}/avg_variability": avg_variability,
                    f"{self.phase}/test_cartography/{dataset_name}/avg_confidence": avg_confidence,
                    f"{self.phase}/test_cartography/{dataset_name}/avg_correctness": avg_correctness,
                    f"{self.phase}/test_cartography/{dataset_name}/num_samples": len(cartography_data)
                })

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:

            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()


            return losses,predictions


    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        self.current_epoch = epoch
        if epoch>=1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1


        #times_per_epoch=4

        test_step = max(1, len(train_data_loader) // times_per_epoch)    # test 10 times per epoch, ensure at least 1
        # Calculate base step count for this epoch (will be incremented per iteration)
        step_cnt_base = (epoch - 1) * len(train_data_loader) + self.global_step_offset

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            # Calculate current step count
            step_cnt = step_cnt_base + iteration + 1
            # Update last_step for tracking
            self.last_step = step_cnt
            
            self.setTrain()
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cuda()

            losses,predictions=self.train_step(data_dict)

            # Store training dynamics for Dataset Cartography
            if 'name' in data_dict and 'label' in data_dict and 'prob' in predictions:
                batch_probs = predictions['prob'].cpu().detach().numpy()
                batch_labels = data_dict['label'].cpu().detach().numpy()
                batch_names = data_dict['name']
                
                # Use 0-indexed epoch for storage
                epoch_idx = epoch - self.config.get('start_epoch', 1)
                
                for idx, sample_path in enumerate(batch_names):
                    if sample_path not in self.training_dynamics:
                        self.training_dynamics[sample_path] = {
                            'path': sample_path,
                            'label': int(batch_labels[idx]),
                            'preds': [],
                            'prob_0': [],
                            'prob_1': []
                        }
                    
                    # Get probability for class 1 (fake)
                    prob_1 = float(batch_probs[idx])
                    prob_0 = 1.0 - prob_1
                    pred = 1 if prob_1 > 0.5 else 0
                    
                    # Ensure lists are long enough for current epoch
                    while len(self.training_dynamics[sample_path]['preds']) < epoch_idx:
                        self.training_dynamics[sample_path]['preds'].append(None)
                        self.training_dynamics[sample_path]['prob_0'].append(None)
                        self.training_dynamics[sample_path]['prob_1'].append(None)
                    
                    # Store for current epoch (append or update)
                    if len(self.training_dynamics[sample_path]['preds']) == epoch_idx:
                        self.training_dynamics[sample_path]['preds'].append(pred)
                        self.training_dynamics[sample_path]['prob_0'].append(prob_0)
                        self.training_dynamics[sample_path]['prob_1'].append(prob_1)
                    else:
                        # Update last entry (handles case where same sample appears multiple times in epoch)
                        self.training_dynamics[sample_path]['preds'][epoch_idx] = pred
                        self.training_dynamics[sample_path]['prob_0'][epoch_idx] = prob_0
                        self.training_dynamics[sample_path]['prob_1'][epoch_idx] = prob_1

            # update learning rate

            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['local_rank']==0:
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                    # WandB-1. loss
                    if self.wandb_run is not None:
                        import wandb
                        wandb.log({f"train/loss/{k}": v_avg}, step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                    # WandB-2. metric
                    if self.wandb_run is not None:
                        import wandb
                        wandb.log({f"train/metric/{k}": v_avg}, step=step_cnt)
                self.logger.info(metric_str)
                
                # Log learning rate to WandB
                if self.wandb_run is not None:
                    import wandb
                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({"train/learning_rate": current_lr}, step=step_cnt)



                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            if (step_cnt+1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp'] ):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                else:
                    test_best_metric = None

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
        return test_best_metric

    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        real_idx = np.where(label == 0)[0]
        fake_idx = np.where(label == 1)[0]
        acc_real = np.count_nonzero(judge[real_idx]) / len(real_idx)
        acc_fake = np.count_nonzero(judge[fake_idx]) / len(fake_idx)

        return acc_real, acc_fake

    def test_one_dataset(self, data_loader, dataset_name=None, epoch=None):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            
            # Store test dynamics for Dataset Cartography (if dataset_name and epoch provided)
            if dataset_name is not None and epoch is not None and 'name' in data_dict:
                # Initialize dataset-specific dynamics dict if not exists
                if dataset_name not in self.test_dynamics:
                    self.test_dynamics[dataset_name] = {}
                
                batch_probs = predictions['prob'].cpu().detach().numpy()
                batch_labels = data_dict['label'].cpu().detach().numpy()
                batch_names = data_dict['name']
                
                # Use 0-indexed epoch for storage
                epoch_idx = epoch - self.config.get('start_epoch', 1)
                
                for idx, sample_path in enumerate(batch_names):
                    if sample_path not in self.test_dynamics[dataset_name]:
                        self.test_dynamics[dataset_name][sample_path] = {
                            'path': sample_path,
                            'label': int(batch_labels[idx]),
                            'preds': [],
                            'prob_0': [],
                            'prob_1': []
                        }
                    
                    # Get probability for class 1 (fake)
                    prob_1 = float(batch_probs[idx])
                    prob_0 = 1.0 - prob_1
                    pred = 1 if prob_1 > 0.5 else 0
                    
                    # Ensure lists are long enough for current epoch
                    while len(self.test_dynamics[dataset_name][sample_path]['preds']) < epoch_idx:
                        self.test_dynamics[dataset_name][sample_path]['preds'].append(None)
                        self.test_dynamics[dataset_name][sample_path]['prob_0'].append(None)
                        self.test_dynamics[dataset_name][sample_path]['prob_1'].append(None)
                    
                    # Store for current epoch (append or update)
                    if len(self.test_dynamics[dataset_name][sample_path]['preds']) == epoch_idx:
                        self.test_dynamics[dataset_name][sample_path]['preds'].append(pred)
                        self.test_dynamics[dataset_name][sample_path]['prob_0'].append(prob_0)
                        self.test_dynamics[dataset_name][sample_path]['prob_1'].append(prob_1)
                    else:
                        # Update last entry (handles case where same sample appears multiple times in epoch)
                        self.test_dynamics[dataset_name][sample_path]['preds'][epoch_idx] = pred
                        self.test_dynamics[dataset_name][sample_path]['prob_0'][epoch_idx] = prob_0
                        self.test_dynamics[dataset_name][sample_path]['prob_1'][epoch_idx] = prob_1
            
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self,epoch,iteration,logging_step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    epoch: {logging_step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=logging_step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
                # WandB-1. test loss
                if self.wandb_run is not None:
                    import wandb
                    wandb.log({f"test/{key}/loss/{k}": v_avg}, step=logging_step)
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    epoch: {logging_step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=logging_step)
            # WandB-2. metric for each dataset
            if self.wandb_run is not None:
                import wandb
                wandb.log({f"test/{key}/{k}": v}, step=logging_step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=logging_step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=logging_step)
            # WandB-3. acc_real and acc_fake
            if self.wandb_run is not None:
                import wandb
                wandb.log({
                    f"test/{key}/acc_real": acc_real,
                    f"test/{key}/acc_fake": acc_fake
                }, step=logging_step)
        self.logger.info(metric_str)
    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset (now with test dynamics tracking)
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(
                test_data_loaders[key], 
                dataset_name=key, 
                epoch=epoch
            )
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
