"""
Extract features from SELFI model and create t-SNE visualization for FaceForensics++
"""
import os
import sys
sys.path.append('./training')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pickle
import yaml
import random
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

import argparse

# Color and label mappings for FaceForensics++
color_map = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
label_dict = {
    0: 'FF_Real',    
    1: 'Deepfakes', 
    2: 'Face2Face', 
    3: 'FaceSwap', 
    4: 'NeuralTextures', 
}

# Mapping from dataset labels to specific labels
label_to_spe = {
    'FF-real': 0,
    'FF-DF': 1,
    'FF-F2F': 2,
    'FF-FS': 3,
    'FF-NT': 4,
}

def get_label_spe_from_path(img_path):
    """
    Extract specific label from image path
    Args:
        img_path: path to image
    Returns:
        specific label (0-4)
    """
    # Check which label is in the path
    for label_name, label_spe in label_to_spe.items():
        if label_name.replace('-', '_').lower() in img_path.lower():
            return label_spe
    
    # If no match found, try to infer from path structure
    if 'original_sequences' in img_path or 'youtube' in img_path:
        return 0  # Real
    elif 'deepfakes' in img_path.lower():
        return 1
    elif 'face2face' in img_path.lower():
        return 2
    elif 'faceswap' in img_path.lower():
        return 3
    elif 'neuraltextures' in img_path.lower():
        return 4
    
    return -1  # Unknown


def tsne_draw(x_transformed, numerical_labels, ax, title='', label_filter=None):
    """
    Draw t-SNE visualization
    Args:
        x_transformed: t-SNE transformed features
        numerical_labels: labels for each sample
        ax: matplotlib axis
        title: plot title
        label_filter: list of labels to include (None for all)
    """
    if label_filter is not None:
        mask = np.isin(numerical_labels, label_filter)
        x_transformed = x_transformed[mask]
        numerical_labels = numerical_labels[mask]
    
    labels = [label_dict[label] for label in numerical_labels]

    tsne_df = pd.DataFrame(x_transformed, columns=['X', 'Y'])
    tsne_df["Targets"] = labels
    tsne_df["NumericTargets"] = numerical_labels
    tsne_df.sort_values(by="NumericTargets", inplace=True)
    
    marker_list = ['*' if label == 0 else 'o' for label in tsne_df["NumericTargets"]]

    for _x, _y, _c, _m in zip(tsne_df['X'], tsne_df['Y'], 
                               [color_map[i] for i in tsne_df["NumericTargets"]], 
                               marker_list):
        ax.scatter(_x, _y, color=_c, s=30, alpha=0.7, marker=_m)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


@torch.no_grad()
def extract_features(model, data_loader, device):
    """
    Extract features and labels from model predictions
    """
    model.eval()
    
    feature_lists = []
    label_spe_lists = []
    img_paths = []
    
    print("Extracting features...")
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
        predictions = model(data_dict, inference=True)
        
        # Collect features (use fused_feat for SELFI model, feat for others)
        if 'fused_feat' in predictions and predictions['fused_feat'] is not None:
            feature_lists.append(predictions['fused_feat'].cpu().detach().numpy())
        else:
            feature_lists.append(predictions['feat'].cpu().detach().numpy())
        
        # Get image paths from data_dict (stored by dataset)
        batch_size = data.shape[0]
        
    # Concatenate all features
    features = np.concatenate(feature_lists, axis=0)
    
    # Get image paths from dataset
    data_dict_full = data_loader.dataset.data_dict
    img_paths = data_dict_full['image']
    
    # Extract specific labels from paths
    label_spe_lists = [get_label_spe_from_path(path) for path in img_paths]
    label_spe_lists = np.array(label_spe_lists)
    
    # Filter out unknown labels
    valid_indices = label_spe_lists != -1
    features = features[valid_indices]
    label_spe_lists = label_spe_lists[valid_indices]
    
    print(f"Extracted {len(features)} features")
    print(f"Label distribution: {np.bincount(label_spe_lists)}")
    
    return features, label_spe_lists


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization for FaceForensics++')
    parser.add_argument('--detector_path', type=str, 
                        default='./training/config/detector_selfi/SELFI_clip.yaml',
                        help='path to detector YAML file')
    parser.add_argument('--weights_path', type=str,
                        required=True,
                        help='path to model weights')
    parser.add_argument('--save_pickle', action='store_true', default=False,
                        help='save extracted features to pickle file')
    parser.add_argument('--output_name', type=str, default='tsne_selfi_ff++',
                        help='output file name (without extension)')
    parser.add_argument('--sample_size', type=int, default=2500,
                        help='number of samples per class for visualization')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    print(f"Loading config from {args.detector_path}")
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    
    # Set test dataset to FaceForensics++
    config['test_dataset'] = 'FaceForensics++'
    config['weights_path'] = args.weights_path
    
    # Init seed
    init_seed(config)
    
    # Set cudnn benchmark
    if config['cudnn']:
        cudnn.benchmark = True
    
    # Prepare data loader
    print("Preparing data loader...")
    test_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='test', 
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set, 
        batch_size=config['test_batchSize'],
        shuffle=False, 
        num_workers=int(config['workers']),
        collate_fn=test_set.collate_fn,
        drop_last=False
    )
    
    # Prepare model
    print(f"Loading model: {config['model_name']}")
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    
    # Load weights
    print(f"Loading weights from {args.weights_path}")
    ckpt = torch.load(args.weights_path, map_location=device)
    ckpt = remove_module_prefix(ckpt)
    model.load_state_dict(ckpt, strict=True)
    print('===> Checkpoint loaded!')
    
    # Extract features
    features, label_spe = extract_features(model, test_data_loader, device)
    
    # Create output folder with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = f'./analysis/{timestamp}_{args.output_name}'
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created output folder: {output_folder}")
    
    # Save features if requested
    if args.save_pickle:
        save_dict = {
            'feat': features,
            'label_spe': label_spe,
        }
        pickle_path = os.path.join(output_folder, 'features.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Features saved to {pickle_path}")
    
    # Sample data for visualization
    print(f"\nSampling {args.sample_size} samples per class...")
    label_0_indices = np.where(label_spe == 0)[0][:args.sample_size]
    other_label_indices = np.where(label_spe != 0)[0]
    
    if len(label_0_indices) == 0:
        print("Warning: No real samples found!")
        return
    
    num_samples = len(label_0_indices)
    if len(other_label_indices) < num_samples:
        print(f"Warning: Only {len(other_label_indices)} fake samples available")
        other_label_indices_sampled = other_label_indices
    else:
        other_label_indices_sampled = np.random.choice(other_label_indices, 
                                                       size=num_samples, 
                                                       replace=False)
    
    sampled_indices = np.concatenate((label_0_indices, other_label_indices_sampled))
    np.random.shuffle(sampled_indices)
    
    feat_sampled = features[sampled_indices]
    label_spe_sampled = label_spe[sampled_indices]
    
    # Reshape features if needed
    if len(feat_sampled.shape) > 2:
        feat_sampled = feat_sampled.reshape((feat_sampled.shape[0], -1))
    
    print(f"Feature shape: {feat_sampled.shape}")
    print(f"Running t-SNE...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=20, random_state=1024, learning_rate=250)
    feat_transformed = tsne.fit_transform(feat_sampled)
    
    # Create legend function
    def create_legend(label_list):
        handles = [plt.Line2D([0], [0], marker='*', color='w', 
                             markerfacecolor=color_map[i], markersize=15) 
                   if i == 0 else 
                   plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=color_map[i], markersize=10) 
                   for i in label_list]
        labels = [label_dict[i] for i in label_list]
        return handles, labels
    
    print("Creating visualizations...")
    
    # 1. All classes (전체)
    print("  - Generating all classes visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    tsne_draw(feat_transformed, label_spe_sampled, ax=ax, 
              title='All Classes', label_filter=None)
    handles, labels = create_legend([0, 1, 2, 3, 4])
    ax.legend(handles, labels, title="Classes", loc="best", fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1_all_classes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Real only (Real만)
    print("  - Generating Real only visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    tsne_draw(feat_transformed, label_spe_sampled, ax=ax, 
              title='Real Only', label_filter=[0])
    handles, labels = create_legend([0])
    ax.legend(handles, labels, title="Classes", loc="best", fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '2_real_only.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Fake only (Fake끼리만)
    print("  - Generating Fake only visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    tsne_draw(feat_transformed, label_spe_sampled, ax=ax, 
              title='Fake Only', label_filter=[1, 2, 3, 4])
    handles, labels = create_legend([1, 2, 3, 4])
    ax.legend(handles, labels, title="Classes", loc="best", fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '3_fake_only.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Each method separately (각 method별로: Real vs each fake)
    print("  - Generating per-method visualizations...")
    method_names = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    
    for idx, (method_label, method_name) in enumerate(zip([1, 2, 3, 4], method_names)):
        tsne_draw(feat_transformed, label_spe_sampled, ax=axes[idx], 
                  title=f'Real vs {method_name}', label_filter=[0, method_label])
        handles, labels = create_legend([0, method_label])
        axes[idx].legend(handles, labels, title="Classes", loc="best", fontsize=14, title_fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '4_per_method.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll visualizations saved to: {output_folder}")
    print("Generated files:")
    print("  - 1_all_classes.png: All 5 classes together")
    print("  - 2_real_only.png: Real samples only")
    print("  - 3_fake_only.png: Fake samples only (4 methods)")
    print("  - 4_per_method.png: Real vs each fake method (2x2 grid)")
    if args.save_pickle:
        print("  - features.pkl: Extracted features and labels")


if __name__ == '__main__':
    main()
