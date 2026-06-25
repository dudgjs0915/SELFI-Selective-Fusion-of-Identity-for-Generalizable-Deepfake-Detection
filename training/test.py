"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
parser.add_argument('--excel', action='store_true', default=False,
                    help='save test results to Excel file')
parser.add_argument('--dataset_json_folder', type=str, default=None, help='path to dataset json folder')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
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
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state_dict keys if present.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    if args.dataset_json_folder:
        config['dataset_json_folder'] = args.dataset_json_folder
    elif config.get('lmdb', False):
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    
    # Enable LoRA if specified in config (needed for loading LoRA-trained models)
    # Check if LoRA should be enabled based on model name or lora_config in YAML
    use_lora = config['model_name'] == 'clip_lora' or 'lora_config' in config
    
    if use_lora:
        if hasattr(model, 'enable_lora'):
            # Get LoRA parameters from config, with defaults if not specified
            lora_config = config.get('lora_config', {})
            lora_r = lora_config.get('r', 8)
            lora_alpha = lora_config.get('alpha', 16)
            lora_dropout = lora_config.get('dropout', 0.1)
            lora_target_modules = lora_config.get('target_modules', None)
            
            print(f'===> Enabling LoRA from config (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})')
            model.enable_lora(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules
            )
            # Move model to device again after adding LoRA layers
            model = model.to(device)
            print('===> LoRA enabled successfully and moved to device')
        else:
            print(f'Warning: Model {config["model_name"]} does not support LoRA')
    
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        ckpt = remove_module_prefix(ckpt)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')
    
    # save results to Excel if --excel flag is set
    if args.excel:
        save_results_to_excel(best_metric, weights_path)

def save_results_to_excel(metrics_dict, weights_path):
    """
    Save test results to Excel file (.xlsx) with two sheets in the log folder.
    Sheet 1: All metrics (dataset, acc, auc, video_auc, eer, ap, ...)
    Sheet 2: Only Dataset and AUC in vertical format
    
    Args:
        metrics_dict: Dictionary containing metrics for all datasets
        weights_path: Path to the weights file (used to determine save location)
    """
    # Create a new workbook
    wb = Workbook()
    
    # === Sheet 1: All metrics ===
    ws1 = wb.active
    ws1.title = "All Metrics"
    
    # Prepare data
    results_data = []
    for dataset_name, metrics in metrics_dict.items():
        row = {'Dataset': dataset_name}
        # Add metrics, excluding 'pred' and 'label' arrays
        for metric_name, metric_value in metrics.items():
            if metric_name not in ['pred', 'label']:
                row[metric_name] = metric_value
        results_data.append(row)
    
    # Get all unique metric names
    metric_names = set()
    for row in results_data:
        metric_names.update(row.keys())
    metric_names.discard('Dataset')
    
    # Define column order
    column_order = ['Dataset', 'acc', 'auc', 'video_auc', 'eer', 'ap']
    available_columns = [col for col in column_order if col in metric_names or col == 'Dataset']
    other_columns = sorted([col for col in metric_names if col not in column_order])
    fieldnames = available_columns + other_columns
    
    # Write header for Sheet 1
    ws1.append(fieldnames)
    # Make header bold
    for cell in ws1[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    
    # Write data rows for Sheet 1
    for row_data in results_data:
        row_values = [row_data.get(field, '') for field in fieldnames]
        ws1.append(row_values)
    
    # Auto-adjust column widths for Sheet 1
    for column in ws1.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws1.column_dimensions[column_letter].width = adjusted_width
    
    # === Sheet 2: Dataset and AUC only (horizontal format) ===
    ws2 = wb.create_sheet(title="AUC Summary")
    
    # Prepare dataset names and AUC values
    dataset_names = []
    auc_values = []
    for dataset_name, metrics in metrics_dict.items():
        dataset_names.append(dataset_name)
        auc_values.append(metrics.get('auc', ''))
    
    # Write first row: Dataset | dataset1 | dataset2 | ...
    ws2.append(['Dataset'] + dataset_names)
    
    # Write second row: AUC | value1 | value2 | ...
    ws2.append(['AUC'] + auc_values)
    
    # Make first column bold
    ws2['A1'].font = Font(bold=True)
    ws2['A2'].font = Font(bold=True)
    ws2['A1'].alignment = Alignment(horizontal='center')
    ws2['A2'].alignment = Alignment(horizontal='center')
    
    # Auto-adjust column widths for Sheet 2
    for column in ws2.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws2.column_dimensions[column_letter].width = adjusted_width
    
    # Determine save path (same directory as weights file)
    if weights_path:
        log_dir = os.path.dirname(weights_path)
    else:
        log_dir = './logs/training'
    
    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    excel_filename = f'test_results_{timestamp}.xlsx'
    excel_path = os.path.join(log_dir, excel_filename)
    
    # Save the workbook
    wb.save(excel_path)
    
    print(f'===> Test results saved to: {excel_path}')
    print(f'     Sheet 1: All Metrics')
    print(f'     Sheet 2: AUC Summary (vertical format)')

if __name__ == '__main__':
    main()
