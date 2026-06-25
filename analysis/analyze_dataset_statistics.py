import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import shutil
from datetime import datetime

def load_and_analyze_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # Check if data is a list or dictionary
    if isinstance(data, list):
        # Data is already in analyzed format (list of samples)
        for sample in data:
            if 'variability' in sample and 'confidence' in sample and 'correctness' in sample:
                # Data already has calculated metrics
                results.append({
                    'path': sample['path'],
                    'label': sample['label'],
                    'correctness': sample['correctness'],
                    'confidence': sample['confidence'],
                    'variability': sample['variability']
                })
            else:
                # Need to calculate metrics from raw data
                path = sample['path']
                label = sample.get('label')
                if label is None:
                    label = 0 if '/youtube/' in path else 1
                
                preds = sample.get('preds', [])
                prob_0 = sample.get('prob_0', [])
                prob_1 = sample.get('prob_1', [])
                
                if len(preds) == 0 or len(prob_0) == 0 or len(prob_1) == 0:
                    continue
                
                # Calculate true label probabilities
                if label == 0:
                    true_label_probs = np.array(prob_0)
                else:
                    true_label_probs = np.array(prob_1)
                
                correctness = np.mean([1 if p == label else 0 for p in preds])
                confidence = np.mean(true_label_probs)
                variability = np.std(true_label_probs)
                
                results.append({
                    'path': path,
                    'label': label,
                    'correctness': correctness,
                    'confidence': confidence,
                    'variability': variability
                })
    else:
        # Data is in dictionary format (old format)
        for path, sample_data in data.items():
            if 'label' not in sample_data or sample_data['label'] is None:
                label = 0 if '/youtube/' in path else 1
            else:
                label = sample_data['label']
            
            epoch_history = sample_data.get('epoch_history', [])
            prob_history = sample_data.get('prob_history', [])
            
            if len(epoch_history) == 0 or len(prob_history) == 0:
                continue
                
            prob_values = np.array([float(p) for p in prob_history])
            if label == 0:
                true_label_probs = 1 - prob_values
            else:
                true_label_probs = prob_values

            correctness = np.mean(epoch_history)
            confidence = np.mean(true_label_probs)
            variability = np.std(true_label_probs)
            
            results.append({
                'path': path,
                'label': label,
                'correctness': correctness,
                'confidence': confidence,
                'variability': variability
            })
    
    return results

def plot_vulnerability_confidence_correctness(results, save_path=None):
    """
    Create scatter plot with variability on x-axis, confidence on y-axis,
    and correctness as color. Real samples use 'x' marker, Fake samples use 'o' marker.
    
    Creates 7 plots:
    1. Combined plot with both Real and Fake samples
    2. Real samples only
    3. Fake samples only (all methods combined)
    4-7. Each of the 4 FaceForensics++ fake methods (Deepfakes, FaceSwap, NeuralTextures, Face2Face)
    """
    # Separate real and fake samples
    real_samples = [r for r in results if r['label'] == 0]
    fake_samples = [r for r in results if r['label'] == 1]
    
    # Separate fake samples by method
    deepfakes_samples = [r for r in fake_samples if '/Deepfakes/' in r['path']]
    faceswap_samples = [r for r in fake_samples if '/FaceSwap/' in r['path']]
    neuraltextures_samples = [r for r in fake_samples if '/NeuralTextures/' in r['path']]
    face2face_samples = [r for r in fake_samples if '/Face2Face/' in r['path']]
    
    real_var = np.array([r['variability'] for r in real_samples])
    real_conf = np.array([r['confidence'] for r in real_samples])
    real_corr = np.array([r['correctness'] for r in real_samples])
    
    fake_var = np.array([r['variability'] for r in fake_samples])
    fake_conf = np.array([r['confidence'] for r in fake_samples])
    fake_corr = np.array([r['correctness'] for r in fake_samples])
    
    all_var = np.array([r['variability'] for r in results])
    print(f"Variability range: [{all_var.min():.4f}, {all_var.max():.4f}]")
    print(f"Sample counts - Real: {len(real_samples)}, Fake: {len(fake_samples)}")
    print(f"  Deepfakes: {len(deepfakes_samples)}, FaceSwap: {len(faceswap_samples)}, "
          f"NeuralTextures: {len(neuraltextures_samples)}, Face2Face: {len(face2face_samples)}")
    
    # Determine save paths for individual plots
    save_dir = None
    combined_path = save_path
    real_only_path = None
    fake_only_path = None
    deepfakes_path = None
    faceswap_path = None
    neuraltextures_path = None
    face2face_path = None
    
    if save_path:
        save_path_obj = Path(save_path)
        save_dir = save_path_obj.parent
        base_name = save_path_obj.stem
        ext = save_path_obj.suffix
        real_only_path = save_dir / f"{base_name}_real_only{ext}"
        fake_only_path = save_dir / f"{base_name}_fake_only{ext}"
        deepfakes_path = save_dir / f"{base_name}_deepfakes{ext}"
        faceswap_path = save_dir / f"{base_name}_faceswap{ext}"
        neuraltextures_path = save_dir / f"{base_name}_neuraltextures{ext}"
        face2face_path = save_dir / f"{base_name}_face2face{ext}"
    
    # Plot 1: Combined (Real + Fake)
    plt.figure(figsize=(12, 8))
    
    # Plot real samples with 'x' marker
    scatter1 = plt.scatter(real_var, real_conf, c=real_corr, 
                          cmap='RdYlGn', alpha=0.6, s=50, marker='x', 
                          vmin=0, vmax=1, label='Real', linewidths=2)
    
    # Plot fake samples with 'o' marker
    scatter2 = plt.scatter(fake_var, fake_conf, c=fake_corr, 
                          cmap='RdYlGn', alpha=0.6, s=20, marker='o', 
                          vmin=0, vmax=1, label='Fake')
    
    plt.xlabel('Variability', fontsize=14)
    plt.ylabel('Confidence', fontsize=14)
    plt.title('Sample Analysis: Variability vs Confidence (Real + Fake)', 
              fontsize=16)
    
    cbar = plt.colorbar(scatter2)
    cbar.set_label('Correctness', fontsize=12)
    
    plt.legend(fontsize=12)
    plt.xlim(-0.025, 0.525)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    if combined_path:
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {combined_path}")
    
    plt.close()
    
    # Plot 2: Real samples only
    plt.figure(figsize=(12, 8))
    
    scatter_real = plt.scatter(real_var, real_conf, c=real_corr, 
                              cmap='RdYlGn', alpha=0.6, s=50, marker='x', 
                              vmin=0, vmax=1, linewidths=2)
    
    plt.xlabel('Variability', fontsize=14)
    plt.ylabel('Confidence', fontsize=14)
    plt.title('Sample Analysis: Variability vs Confidence (Real Only)', 
              fontsize=16)
    
    cbar = plt.colorbar(scatter_real)
    cbar.set_label('Correctness', fontsize=12)
    
    plt.xlim(-0.025, 0.525)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    if real_only_path:
        plt.savefig(real_only_path, dpi=300, bbox_inches='tight')
        print(f"Real-only plot saved to {real_only_path}")
    
    plt.close()
    
    # Plot 3: Fake samples only (all methods)
    plt.figure(figsize=(12, 8))
    
    scatter_fake = plt.scatter(fake_var, fake_conf, c=fake_corr, 
                              cmap='RdYlGn', alpha=0.6, s=20, marker='o', 
                              vmin=0, vmax=1)
    
    plt.xlabel('Variability', fontsize=14)
    plt.ylabel('Confidence', fontsize=14)
    plt.title('Sample Analysis: Variability vs Confidence (Fake Only - All Methods)', 
              fontsize=16)
    
    cbar = plt.colorbar(scatter_fake)
    cbar.set_label('Correctness', fontsize=12)
    
    plt.xlim(-0.025, 0.525)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    if fake_only_path:
        plt.savefig(fake_only_path, dpi=300, bbox_inches='tight')
        print(f"Fake-only plot saved to {fake_only_path}")
    
    plt.close()
    
    # Plot 4-7: Individual fake methods
    fake_methods = [
        (deepfakes_samples, deepfakes_path, 'Deepfakes'),
        (faceswap_samples, faceswap_path, 'FaceSwap'),
        (neuraltextures_samples, neuraltextures_path, 'NeuralTextures'),
        (face2face_samples, face2face_path, 'Face2Face')
    ]
    
    for method_samples, method_path, method_name in fake_methods:
        if len(method_samples) == 0:
            print(f"Warning: No samples found for {method_name}, skipping plot")
            continue
        
        method_var = np.array([r['variability'] for r in method_samples])
        method_conf = np.array([r['confidence'] for r in method_samples])
        method_corr = np.array([r['correctness'] for r in method_samples])
        
        plt.figure(figsize=(12, 8))
        
        scatter_method = plt.scatter(method_var, method_conf, c=method_corr, 
                                    cmap='RdYlGn', alpha=0.6, s=20, marker='o', 
                                    vmin=0, vmax=1)
        
        plt.xlabel('Variability', fontsize=14)
        plt.ylabel('Confidence', fontsize=14)
        plt.title(f'Sample Analysis: Variability vs Confidence ({method_name})', 
                  fontsize=16)
        
        cbar = plt.colorbar(scatter_method)
        cbar.set_label('Correctness', fontsize=12)
        
        plt.xlim(-0.025, 0.525)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        
        if method_path:
            plt.savefig(method_path, dpi=300, bbox_inches='tight')
            print(f"{method_name} plot saved to {method_path}")
        
        plt.close()

def parse_path_info(path):
    """
    Parse file path to extract dataset information.
    
    Returns:
        dict with keys: 'dataset_type' (FF-real, FF-DF, FF-FS, FF-NT, FF-F2F),
                       'split' (train/test/val), 'compression' (c23/c40),
                       'video_id', 'frame'
    """
    import re
    
    # Determine if real or fake
    if '/youtube/' in path:
        dataset_type = 'FF-real'
        # Path: .../original_sequences/youtube/c23/frames/294/058.png
        match = re.search(r'/youtube/(c\d+)/frames/(\d+)/(\d+)\.png', path)
        if match:
            compression = match.group(1)
            video_id = match.group(2)
            frame = match.group(3)
    else:
        # Path: .../manipulated_sequences/Deepfakes/c23/frames/396_272/067.png
        method_map = {
            'Deepfakes': 'FF-DF',
            'FaceSwap': 'FF-FS',
            'Face2Face': 'FF-F2F',
            'NeuralTextures': 'FF-NT'
        }
        
        for method_name, dataset_name in method_map.items():
            if f'/{method_name}/' in path:
                dataset_type = dataset_name
                match = re.search(rf'/{method_name}/(c\d+)/frames/([^/]+)/(\d+)\.png', path)
                if match:
                    compression = match.group(1)
                    video_id = match.group(2)
                    frame = match.group(3)
                break
        else:
            return None
    
    # For train data, split is 'train'
    split = 'train'
    
    return {
        'dataset_type': dataset_type,
        'split': split,
        'compression': compression,
        'video_id': video_id,
        'frame': frame
    }

def extract_top_k_percent_by_variability(results, k_percent=10, save_path=None, verbose=True, original_json_path=None):
    """
    Extract top k% samples with highest variability and save as FaceForensics++ JSON structure.
    
    Args:
        results: List of result dictionaries with 'variability' field
        k_percent: Percentage of top samples to extract (default: 10)
        save_path: Path to save JSON file (optional)
        verbose: Whether to print detailed statistics (default: True)
        original_json_path: Path to original FaceForensics++.json to copy test/val splits (optional)
    
    Returns:
        List of top k% samples sorted by variability (descending)
    """
    # Sort by variability in descending order
    sorted_results = sorted(results, key=lambda x: x['variability'], reverse=True)
    
    # Calculate number of samples for top k%
    k_samples = int(len(sorted_results) * k_percent / 100)
    top_k_results = sorted_results[:k_samples]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TOP {k_percent}% SAMPLES BY VARIABILITY")
        print(f"{'='*60}")
        print(f"Total samples: {len(results)}")
        print(f"Top {k_percent}% samples: {k_samples}")
        if k_samples > 0:
            print(f"Variability range in top {k_percent}%: [{top_k_results[-1]['variability']:.4f}, {top_k_results[0]['variability']:.4f}]")
    
    # Create FaceForensics++ structure
    ff_structure = {
        "FaceForensics++": {}
    }
    
    # Group by dataset_type -> split -> compression -> video_id -> frames
    for r in top_k_results:
        path_info = parse_path_info(r['path'])
        if path_info is None:
            continue
        
        dataset_type = path_info['dataset_type']
        split = path_info['split']
        compression = path_info['compression']
        video_id = path_info['video_id']
        
        # Initialize nested structure
        if dataset_type not in ff_structure["FaceForensics++"]:
            ff_structure["FaceForensics++"][dataset_type] = {}
        if split not in ff_structure["FaceForensics++"][dataset_type]:
            ff_structure["FaceForensics++"][dataset_type][split] = {}
        if compression not in ff_structure["FaceForensics++"][dataset_type][split]:
            ff_structure["FaceForensics++"][dataset_type][split][compression] = {}
        if video_id not in ff_structure["FaceForensics++"][dataset_type][split][compression]:
            ff_structure["FaceForensics++"][dataset_type][split][compression][video_id] = {
                'label': dataset_type,
                'frames': []
            }
        
        # Add frame path
        ff_structure["FaceForensics++"][dataset_type][split][compression][video_id]['frames'].append(r['path'])
    
    # Print statistics by dataset type
    if verbose:
        print(f"\nDataset distribution:")
        for dataset_type in ff_structure["FaceForensics++"]:
            for split in ff_structure["FaceForensics++"][dataset_type]:
                for compression in ff_structure["FaceForensics++"][dataset_type][split]:
                    video_count = len(ff_structure["FaceForensics++"][dataset_type][split][compression])
                    frame_count = sum(len(v['frames']) for v in ff_structure["FaceForensics++"][dataset_type][split][compression].values())
                    print(f"  {dataset_type}/{split}/{compression}: {video_count} videos, {frame_count} frames")
    
    # Add test and val splits from original JSON if provided
    if original_json_path and Path(original_json_path).exists():
        if verbose:
            print(f"\nAdding test/val splits from original JSON: {original_json_path}")
        with open(original_json_path, 'r') as f:
            original_data = json.load(f)
        
        if 'FaceForensics++' in original_data:
            for dataset_type in original_data['FaceForensics++']:
                # Initialize dataset_type if not exists
                if dataset_type not in ff_structure['FaceForensics++']:
                    ff_structure['FaceForensics++'][dataset_type] = {}
                
                # Copy test and val splits
                for split in ['test', 'val']:
                    if split in original_data['FaceForensics++'][dataset_type]:
                        ff_structure['FaceForensics++'][dataset_type][split] = original_data['FaceForensics++'][dataset_type][split]
                        if verbose:
                            # Count videos and frames in copied splits
                            for compression in ff_structure['FaceForensics++'][dataset_type][split]:
                                video_count = len(ff_structure['FaceForensics++'][dataset_type][split][compression])
                                frame_count = sum(len(v['frames']) for v in ff_structure['FaceForensics++'][dataset_type][split][compression].values())
                                print(f"  {dataset_type}/{split}/{compression}: {video_count} videos, {frame_count} frames (from original)")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(ff_structure, f, indent=2)
        if verbose:
            print(f"\nTop {k_percent}% samples saved to: {save_path}")
    
    if verbose:
        print(f"{'='*60}\n")
    
    return top_k_results

def extract_top_k_percent_by_variability_separate(results, k_percent=10, save_path=None, verbose=True, original_json_path=None):
    """
    Extract top k% samples with highest variability separately for Real and Fake samples.
    
    Args:
        results: List of result dictionaries with 'variability' and 'label' fields
        k_percent: Percentage of top samples to extract from each class (default: 10)
        save_path: Path to save JSON file (optional)
        verbose: Whether to print detailed statistics (default: True)
        original_json_path: Path to original FaceForensics++.json to copy test/val splits (optional)
    
    Returns:
        List of top k% samples (Real + Fake) sorted by variability (descending)
    """
    # Separate Real and Fake samples
    real_samples = [r for r in results if r['label'] == 0]
    fake_samples = [r for r in results if r['label'] == 1]
    
    # Sort each by variability
    sorted_real = sorted(real_samples, key=lambda x: x['variability'], reverse=True)
    sorted_fake = sorted(fake_samples, key=lambda x: x['variability'], reverse=True)
    
    # Calculate number of samples for top k%
    k_real = int(len(sorted_real) * k_percent / 100)
    k_fake = int(len(sorted_fake) * k_percent / 100)
    
    top_k_real = sorted_real[:k_real]
    top_k_fake = sorted_fake[:k_fake]
    top_k_results = top_k_real + top_k_fake
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TOP {k_percent}% SAMPLES BY VARIABILITY (SEPARATE REAL/FAKE)")
        print(f"{'='*60}")
        print(f"Total Real samples: {len(real_samples)} -> Top {k_percent}%: {k_real}")
        if k_real > 0:
            print(f"  Real variability range: [{top_k_real[-1]['variability']:.4f}, {top_k_real[0]['variability']:.4f}]")
        print(f"Total Fake samples: {len(fake_samples)} -> Top {k_percent}%: {k_fake}")
        if k_fake > 0:
            print(f"  Fake variability range: [{top_k_fake[-1]['variability']:.4f}, {top_k_fake[0]['variability']:.4f}]")
        print(f"Total selected: {len(top_k_results)}")
    
    # Create FaceForensics++ structure
    ff_structure = {"FaceForensics++": {}}
    
    for r in top_k_results:
        path_info = parse_path_info(r['path'])
        if path_info is None:
            continue
        
        dataset_type = path_info['dataset_type']
        split = path_info['split']
        compression = path_info['compression']
        video_id = path_info['video_id']
        
        if dataset_type not in ff_structure["FaceForensics++"]:
            ff_structure["FaceForensics++"][dataset_type] = {}
        if split not in ff_structure["FaceForensics++"][dataset_type]:
            ff_structure["FaceForensics++"][dataset_type][split] = {}
        if compression not in ff_structure["FaceForensics++"][dataset_type][split]:
            ff_structure["FaceForensics++"][dataset_type][split][compression] = {}
        if video_id not in ff_structure["FaceForensics++"][dataset_type][split][compression]:
            ff_structure["FaceForensics++"][dataset_type][split][compression][video_id] = {
                'label': dataset_type,
                'frames': []
            }
        
        ff_structure["FaceForensics++"][dataset_type][split][compression][video_id]['frames'].append(r['path'])
    
    if verbose:
        print(f"\nDataset distribution:")
        for dataset_type in ff_structure["FaceForensics++"]:
            for split in ff_structure["FaceForensics++"][dataset_type]:
                for compression in ff_structure["FaceForensics++"][dataset_type][split]:
                    video_count = len(ff_structure["FaceForensics++"][dataset_type][split][compression])
                    frame_count = sum(len(v['frames']) for v in ff_structure["FaceForensics++"][dataset_type][split][compression].values())
                    print(f"  {dataset_type}/{split}/{compression}: {video_count} videos, {frame_count} frames")
    
    # Add test and val splits
    if original_json_path and Path(original_json_path).exists():
        with open(original_json_path, 'r') as f:
            original_data = json.load(f)
        
        if 'FaceForensics++' in original_data:
            for dataset_type in original_data['FaceForensics++']:
                if dataset_type not in ff_structure['FaceForensics++']:
                    ff_structure['FaceForensics++'][dataset_type] = {}
                
                for split in ['test', 'val']:
                    if split in original_data['FaceForensics++'][dataset_type]:
                        ff_structure['FaceForensics++'][dataset_type][split] = original_data['FaceForensics++'][dataset_type][split]
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(ff_structure, f, indent=2)
        if verbose:
            print(f"\nSaved to: {save_path}")
    
    if verbose:
        print(f"{'='*60}\n")
    
    return top_k_results

def extract_top_k_percent_by_confidence_separate(results, k_percent=20, save_path=None, verbose=True, original_json_path=None):
    """
    Extract top k% samples with highest confidence separately for Real and Fake samples.
    
    Args:
        results: List of result dictionaries with 'confidence' and 'label' fields
        k_percent: Percentage of top samples to extract from each class (default: 20)
        save_path: Path to save JSON file (optional)
        verbose: Whether to print detailed statistics (default: True)
        original_json_path: Path to original FaceForensics++.json to copy test/val splits (optional)
    
    Returns:
        List of top k% samples (Real + Fake) sorted by confidence (descending)
    """
    # Separate Real and Fake samples
    real_samples = [r for r in results if r['label'] == 0]
    fake_samples = [r for r in results if r['label'] == 1]
    
    # Sort each by confidence (descending)
    sorted_real = sorted(real_samples, key=lambda x: x['confidence'], reverse=True)
    sorted_fake = sorted(fake_samples, key=lambda x: x['confidence'], reverse=True)
    
    # Calculate number of samples for top k%
    k_real = int(len(sorted_real) * k_percent / 100)
    k_fake = int(len(sorted_fake) * k_percent / 100)
    
    top_k_real = sorted_real[:k_real]
    top_k_fake = sorted_fake[:k_fake]
    top_k_results = top_k_real + top_k_fake
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TOP {k_percent}% SAMPLES BY CONFIDENCE (SEPARATE REAL/FAKE)")
        print(f"{'='*60}")
        print(f"Total Real samples: {len(real_samples)} -> Top {k_percent}%: {k_real}")
        if k_real > 0:
            print(f"  Real confidence range: [{top_k_real[-1]['confidence']:.4f}, {top_k_real[0]['confidence']:.4f}]")
        print(f"Total Fake samples: {len(fake_samples)} -> Top {k_percent}%: {k_fake}")
        if k_fake > 0:
            print(f"  Fake confidence range: [{top_k_fake[-1]['confidence']:.4f}, {top_k_fake[0]['confidence']:.4f}]")
        print(f"Total selected: {len(top_k_results)}")
    
    # Create FaceForensics++ structure
    ff_structure = {"FaceForensics++": {}}
    
    for r in top_k_results:
        path_info = parse_path_info(r['path'])
        if path_info is None:
            continue
        
        dataset_type = path_info['dataset_type']
        split = path_info['split']
        compression = path_info['compression']
        video_id = path_info['video_id']
        
        if dataset_type not in ff_structure["FaceForensics++"]:
            ff_structure["FaceForensics++"][dataset_type] = {}
        if split not in ff_structure["FaceForensics++"][dataset_type]:
            ff_structure["FaceForensics++"][dataset_type][split] = {}
        if compression not in ff_structure["FaceForensics++"][dataset_type][split]:
            ff_structure["FaceForensics++"][dataset_type][split][compression] = {}
        if video_id not in ff_structure["FaceForensics++"][dataset_type][split][compression]:
            ff_structure["FaceForensics++"][dataset_type][split][compression][video_id] = {
                'label': dataset_type,
                'frames': []
            }
        
        ff_structure["FaceForensics++"][dataset_type][split][compression][video_id]['frames'].append(r['path'])
    
    if verbose:
        print(f"\nDataset distribution:")
        for dataset_type in ff_structure["FaceForensics++"]:
            for split in ff_structure["FaceForensics++"][dataset_type]:
                for compression in ff_structure["FaceForensics++"][dataset_type][split]:
                    video_count = len(ff_structure["FaceForensics++"][dataset_type][split][compression])
                    frame_count = sum(len(v['frames']) for v in ff_structure["FaceForensics++"][dataset_type][split][compression].values())
                    print(f"  {dataset_type}/{split}/{compression}: {video_count} videos, {frame_count} frames")
    
    # Add test and val splits
    if original_json_path and Path(original_json_path).exists():
        with open(original_json_path, 'r') as f:
            original_data = json.load(f)
        
        if 'FaceForensics++' in original_data:
            for dataset_type in original_data['FaceForensics++']:
                if dataset_type not in ff_structure['FaceForensics++']:
                    ff_structure['FaceForensics++'][dataset_type] = {}
                
                for split in ['test', 'val']:
                    if split in original_data['FaceForensics++'][dataset_type]:
                        ff_structure['FaceForensics++'][dataset_type][split] = original_data['FaceForensics++'][dataset_type][split]
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(ff_structure, f, indent=2)
        if verbose:
            print(f"\nSaved to: {save_path}")
    
    if verbose:
        print(f"{'='*60}\n")
    
    return top_k_results


def extract_bottom_k_percent_by_variability_separate(results, k_percent=20, save_path=None, verbose=True, original_json_path=None):
    """
    Extract bottom k% samples with lowest variability separately for Real and Fake samples.
    
    Args:
        results: List of result dictionaries with 'variability' and 'label' fields
        k_percent: Percentage of bottom samples to extract from each class (default: 20)
        save_path: Path to save JSON file (optional)
        verbose: Whether to print detailed statistics (default: True)
        original_json_path: Path to original FaceForensics++.json to copy test/val splits (optional)
    
    Returns:
        List of bottom k% samples (Real + Fake) sorted by variability (ascending)
    """
    # Separate Real and Fake samples
    real_samples = [r for r in results if r['label'] == 0]
    fake_samples = [r for r in results if r['label'] == 1]
    
    # Sort each by variability (ascending)
    sorted_real = sorted(real_samples, key=lambda x: x['variability'])
    sorted_fake = sorted(fake_samples, key=lambda x: x['variability'])
    
    # Calculate number of samples for bottom k%
    k_real = int(len(sorted_real) * k_percent / 100)
    k_fake = int(len(sorted_fake) * k_percent / 100)
    
    bottom_k_real = sorted_real[:k_real]
    bottom_k_fake = sorted_fake[:k_fake]
    bottom_k_results = bottom_k_real + bottom_k_fake
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BOTTOM {k_percent}% SAMPLES BY VARIABILITY (SEPARATE REAL/FAKE)")
        print(f"{'='*60}")
        print(f"Total Real samples: {len(real_samples)} -> Bottom {k_percent}%: {k_real}")
        if k_real > 0:
            print(f"  Real variability range: [{bottom_k_real[0]['variability']:.4f}, {bottom_k_real[-1]['variability']:.4f}]")
        print(f"Total Fake samples: {len(fake_samples)} -> Bottom {k_percent}%: {k_fake}")
        if k_fake > 0:
            print(f"  Fake variability range: [{bottom_k_fake[0]['variability']:.4f}, {bottom_k_fake[-1]['variability']:.4f}]")
        print(f"Total selected: {len(bottom_k_results)}")
    
    # Create FaceForensics++ structure
    ff_structure = {"FaceForensics++": {}}
    
    for r in bottom_k_results:
        path_info = parse_path_info(r['path'])
        if path_info is None:
            continue
        
        dataset_type = path_info['dataset_type']
        split = path_info['split']
        compression = path_info['compression']
        video_id = path_info['video_id']
        
        if dataset_type not in ff_structure["FaceForensics++"]:
            ff_structure["FaceForensics++"][dataset_type] = {}
        if split not in ff_structure["FaceForensics++"][dataset_type]:
            ff_structure["FaceForensics++"][dataset_type][split] = {}
        if compression not in ff_structure["FaceForensics++"][dataset_type][split]:
            ff_structure["FaceForensics++"][dataset_type][split][compression] = {}
        if video_id not in ff_structure["FaceForensics++"][dataset_type][split][compression]:
            ff_structure["FaceForensics++"][dataset_type][split][compression][video_id] = {
                'label': dataset_type,
                'frames': []
            }
        
        ff_structure["FaceForensics++"][dataset_type][split][compression][video_id]['frames'].append(r['path'])
    
    if verbose:
        print(f"\nDataset distribution:")
        for dataset_type in ff_structure["FaceForensics++"]:
            for split in ff_structure["FaceForensics++"][dataset_type]:
                for compression in ff_structure["FaceForensics++"][dataset_type][split]:
                    video_count = len(ff_structure["FaceForensics++"][dataset_type][split][compression])
                    frame_count = sum(len(v['frames']) for v in ff_structure["FaceForensics++"][dataset_type][split][compression].values())
                    print(f"  {dataset_type}/{split}/{compression}: {video_count} videos, {frame_count} frames")
    
    # Add test and val splits
    if original_json_path and Path(original_json_path).exists():
        with open(original_json_path, 'r') as f:
            original_data = json.load(f)
        
        if 'FaceForensics++' in original_data:
            for dataset_type in original_data['FaceForensics++']:
                if dataset_type not in ff_structure['FaceForensics++']:
                    ff_structure['FaceForensics++'][dataset_type] = {}
                
                for split in ['test', 'val']:
                    if split in original_data['FaceForensics++'][dataset_type]:
                        ff_structure['FaceForensics++'][dataset_type][split] = original_data['FaceForensics++'][dataset_type][split]
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(ff_structure, f, indent=2)
        if verbose:
            print(f"\nSaved to: {save_path}")
    
    if verbose:
        print(f"{'='*60}\n")
    
    return bottom_k_results


def extract_bottom_k_percent_by_confidence_separate(results, k_percent=20, save_path=None, verbose=True, original_json_path=None):
    """
    Extract bottom k% samples with lowest confidence separately for Real and Fake samples.
    
    Args:
        results: List of result dictionaries with 'confidence' and 'label' fields
        k_percent: Percentage of bottom samples to extract from each class (default: 20)
        save_path: Path to save JSON file (optional)
        verbose: Whether to print detailed statistics (default: True)
        original_json_path: Path to original FaceForensics++.json to copy test/val splits (optional)
    
    Returns:
        List of bottom k% samples (Real + Fake) sorted by confidence (ascending)
    """
    # Separate Real and Fake samples
    real_samples = [r for r in results if r['label'] == 0]
    fake_samples = [r for r in results if r['label'] == 1]
    
    # Sort each by confidence (ascending)
    sorted_real = sorted(real_samples, key=lambda x: x['confidence'])
    sorted_fake = sorted(fake_samples, key=lambda x: x['confidence'])
    
    # Calculate number of samples for bottom k%
    k_real = int(len(sorted_real) * k_percent / 100)
    k_fake = int(len(sorted_fake) * k_percent / 100)
    
    bottom_k_real = sorted_real[:k_real]
    bottom_k_fake = sorted_fake[:k_fake]
    bottom_k_results = bottom_k_real + bottom_k_fake
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BOTTOM {k_percent}% SAMPLES BY CONFIDENCE (SEPARATE REAL/FAKE)")
        print(f"{'='*60}")
        print(f"Total Real samples: {len(real_samples)} -> Bottom {k_percent}%: {k_real}")
        if k_real > 0:
            print(f"  Real confidence range: [{bottom_k_real[0]['confidence']:.4f}, {bottom_k_real[-1]['confidence']:.4f}]")
        print(f"Total Fake samples: {len(fake_samples)} -> Bottom {k_percent}%: {k_fake}")
        if k_fake > 0:
            print(f"  Fake confidence range: [{bottom_k_fake[0]['confidence']:.4f}, {bottom_k_fake[-1]['confidence']:.4f}]")
        print(f"Total selected: {len(bottom_k_results)}")
    
    # Create FaceForensics++ structure
    ff_structure = {"FaceForensics++": {}}
    
    for r in bottom_k_results:
        path_info = parse_path_info(r['path'])
        if path_info is None:
            continue
        
        dataset_type = path_info['dataset_type']
        split = path_info['split']
        compression = path_info['compression']
        video_id = path_info['video_id']
        
        if dataset_type not in ff_structure["FaceForensics++"]:
            ff_structure["FaceForensics++"][dataset_type] = {}
        if split not in ff_structure["FaceForensics++"][dataset_type]:
            ff_structure["FaceForensics++"][dataset_type][split] = {}
        if compression not in ff_structure["FaceForensics++"][dataset_type][split]:
            ff_structure["FaceForensics++"][dataset_type][split][compression] = {}
        if video_id not in ff_structure["FaceForensics++"][dataset_type][split][compression]:
            ff_structure["FaceForensics++"][dataset_type][split][compression][video_id] = {
                'label': dataset_type,
                'frames': []
            }
        
        ff_structure["FaceForensics++"][dataset_type][split][compression][video_id]['frames'].append(r['path'])
    
    if verbose:
        print(f"\nDataset distribution:")
        for dataset_type in ff_structure["FaceForensics++"]:
            for split in ff_structure["FaceForensics++"][dataset_type]:
                for compression in ff_structure["FaceForensics++"][dataset_type][split]:
                    video_count = len(ff_structure["FaceForensics++"][dataset_type][split][compression])
                    frame_count = sum(len(v['frames']) for v in ff_structure["FaceForensics++"][dataset_type][split][compression].values())
                    print(f"  {dataset_type}/{split}/{compression}: {video_count} videos, {frame_count} frames")
    
    # Add test and val splits
    if original_json_path and Path(original_json_path).exists():
        with open(original_json_path, 'r') as f:
            original_data = json.load(f)
        
        if 'FaceForensics++' in original_data:
            for dataset_type in original_data['FaceForensics++']:
                if dataset_type not in ff_structure['FaceForensics++']:
                    ff_structure['FaceForensics++'][dataset_type] = {}
                
                for split in ['test', 'val']:
                    if split in original_data['FaceForensics++'][dataset_type]:
                        ff_structure['FaceForensics++'][dataset_type][split] = original_data['FaceForensics++'][dataset_type][split]
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(ff_structure, f, indent=2)
        if verbose:
            print(f"\nSaved to: {save_path}")
    
    if verbose:
        print(f"{'='*60}\n")
    
    return bottom_k_results

def print_statistics(results):
    """Print summary statistics"""
    print("="*60)
    print("SAMPLE TRACKING STATISTICS")
    print("="*60)
    print(f"Total samples: {len(results)}")
    
    correctnesses = [r['correctness'] for r in results]
    variabilities = [r['variability'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    print(f"\nCorrectness: mean={np.mean(correctnesses):.3f}, std={np.std(correctnesses):.3f}, min={np.min(correctnesses):.3f}, max={np.max(correctnesses):.3f}")
    print(f"Variability: mean={np.mean(variabilities):.3f}, std={np.std(variabilities):.3f}, min={np.min(variabilities):.3f}, max={np.max(variabilities):.3f}")
    print(f"Confidence: mean={np.mean(confidences):.3f}, std={np.std(confidences):.3f}, min={np.min(confidences):.3f}, max={np.max(confidences):.3f}")
    
    # Categorize samples
    unforgettable = [r for r in results if r['correctness'] == 1.0 and r['variability'] < 0.01]
    forgettable = [r for r in results if r['variability'] > 0.1]
    always_wrong = [r for r in results if r['correctness'] == 0]
    
    print(f"\nUnforgettable (correctness=1.0, variability<0.01): {len(unforgettable)} ({len(unforgettable)/len(results)*100:.1f}%)")
    print(f"Forgettable (variability>0.1): {len(forgettable)} ({len(forgettable)/len(results)*100:.1f}%)")
    print(f"Always wrong: {len(always_wrong)} ({len(always_wrong)/len(results)*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze dataset statistics from sample tracking JSON')
    parser.add_argument('json_path', type=str, help='Path to the sample_tracking_statistics.json file')
    args = parser.parse_args()
    
    json_path = args.json_path
    
    # Create cartography_results folder structure
    script_dir = Path(__file__).parent
    cartography_base = script_dir / "cartography_results"
    cartography_base.mkdir(exist_ok=True)
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_folder = cartography_base / timestamp
    result_folder.mkdir(exist_ok=True)
    
    # Create train_json subfolder
    train_json_folder = result_folder / "train_json"
    train_json_folder.mkdir(exist_ok=True)
    
    print(f"Results will be saved to: {result_folder}")
    
    # Copy input JSON file
    input_json_copy = result_folder / "original_sample_tracking.json"
    shutil.copy2(json_path, input_json_copy)
    print(f"Input JSON copied to: {input_json_copy}")
    
    # Load and analyze
    print("Loading JSON file...")
    results = load_and_analyze_json(json_path)
    
    # Print statistics
    print_statistics(results)
    
    # Save sorted results by variability
    sorted_results = sorted(results, key=lambda x: x['variability'], reverse=True)
    sorted_json_path = result_folder / "sorted_by_variability.json"
    with open(sorted_json_path, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    print(f"Sorted results saved to: {sorted_json_path}")
    
    # Find original FaceForensics++.json
    script_dir = Path(__file__).parent
    original_json_path = script_dir.parent / "preprocessing" / "dataset_json" / "FaceForensics++.json"
    if not original_json_path.exists():
        print(f"Warning: Original FaceForensics++.json not found at {original_json_path}")
        print("Test/val splits will not be included in the generated JSON files.")
        original_json_path = None
    else:
        print(f"Found original JSON at: {original_json_path}")
    
    # Create train_json files with 4 scenarios
    percentiles = [20, 30, 40, 50, 60, 70, 80]
    
    # 1. Top k% Variability (Real/Fake separate)
    print("\n1. Creating Top k% Variability datasets (Real/Fake separate)...")
    for k_percent in percentiles:
        dataset_path = train_json_folder / f"variability_top{k_percent}.json"
        extract_top_k_percent_by_variability_separate(results, k_percent=k_percent, save_path=dataset_path, 
                                                      verbose=False, original_json_path=original_json_path)
        print(f"  Created: variability_top{k_percent}.json")
    
    # 2. Top k% Confidence (Real/Fake separate)
    print("\n2. Creating Top k% Confidence datasets (Real/Fake separate)...")
    for k_percent in percentiles:
        dataset_path = train_json_folder / f"confidence_top{k_percent}.json"
        extract_top_k_percent_by_confidence_separate(results, k_percent=k_percent, save_path=dataset_path,
                                                     verbose=False, original_json_path=original_json_path)
        print(f"  Created: confidence_top{k_percent}.json")
    
    # 3. Bottom k% Variability (Real/Fake separate)
    print("\n3. Creating Bottom k% Variability datasets (Real/Fake separate)...")
    for k_percent in percentiles:
        dataset_path = train_json_folder / f"variability_bottom{k_percent}.json"
        extract_bottom_k_percent_by_variability_separate(results, k_percent=k_percent, save_path=dataset_path,
                                                         verbose=False, original_json_path=original_json_path)
        print(f"  Created: variability_bottom{k_percent}.json")
    
    # 4. Bottom k% Confidence (Real/Fake separate)
    print("\n4. Creating Bottom k% Confidence datasets (Real/Fake separate)...")
    for k_percent in percentiles:
        dataset_path = train_json_folder / f"confidence_bottom{k_percent}.json"
        extract_bottom_k_percent_by_confidence_separate(results, k_percent=k_percent, save_path=dataset_path,
                                                        verbose=False, original_json_path=original_json_path)
        print(f"  Created: confidence_bottom{k_percent}.json")
    
    # Create plot
    print("\nGenerating plot...")
    plot_path = result_folder / "variability_confidence_plot.png"
    plot_vulnerability_confidence_correctness(results, plot_path)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"All results saved to: {result_folder}")
    print(f"  - Original JSON: original_sample_tracking.json")
    print(f"  - Sorted results: sorted_by_variability.json")
    print(f"  - Plots: 7 visualization plots")
    print(f"    * variability_confidence_plot.png (Real + Fake)")
    print(f"    * variability_confidence_plot_real_only.png")
    print(f"    * variability_confidence_plot_fake_only.png")
    print(f"    * variability_confidence_plot_deepfakes.png")
    print(f"    * variability_confidence_plot_faceswap.png")
    print(f"    * variability_confidence_plot_neuraltextures.png")
    print(f"    * variability_confidence_plot_face2face.png")
    print(f"  - Train datasets: train_json/ (28 files total)")
    print(f"    * Top k% variability: 7 files (20%-80% by 10%)")
    print(f"    * Top k% confidence: 7 files (20%-80% by 10%)")
    print(f"    * Bottom k% variability: 7 files (20%-80% by 10%)")
    print(f"    * Bottom k% confidence: 7 files (20%-80% by 10%)")
    print(f"{'='*60}")
