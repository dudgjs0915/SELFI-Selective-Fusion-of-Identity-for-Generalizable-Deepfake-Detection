#!/usr/bin/env python3
"""
Create dataset folders from cartography results.

This script takes a train_json folder path and creates corresponding dataset_json folders
in the preprocessing directory. Each JSON file in the train_json folder will get its own
dataset folder with the filtered FaceForensics++ data and optional test datasets.

Usage:
    python create_dataset_folders.py <train_json_folder_path> [--test-datasets DATASET1 DATASET2 ...]

Example:
    python create_dataset_folders.py /path/to/cartography_results/XXX/train_json --test-datasets Celeb-DF-v2 DFDCP
"""

import argparse
import json
import shutil
from pathlib import Path


def convert_filename_to_folder_name(json_filename):
    """
    Convert JSON filename to folder name.
    
    Example:
        variability_bottom20.json -> dataset_json_variability_bottom_20
        confidence_top30.json -> dataset_json_confidence_top_30
    """
    # Remove .json extension
    name_without_ext = json_filename.replace('.json', '')
    
    # Add underscores between letters and numbers
    # e.g., bottom20 -> bottom_20
    import re
    name_with_underscores = re.sub(r'([a-z])(\d)', r'\1_\2', name_without_ext)
    
    # Add dataset_json_ prefix
    folder_name = f"dataset_json_{name_with_underscores}"
    
    return folder_name


def create_dataset_folders(train_json_folder, test_datasets=None, preprocessing_base=None):
    """
    Create dataset folders from train_json files.
    
    Args:
        train_json_folder: Path to the train_json folder containing filtered JSON files
        test_datasets: List of test dataset names to copy (e.g., ['Celeb-DF-v2', 'DFDCP'])
        preprocessing_base: Base path for preprocessing folder (default: auto-detect)
    """
    train_json_path = Path(train_json_folder)
    
    if not train_json_path.exists():
        print(f"Error: train_json folder not found: {train_json_path}")
        return
    
    if not train_json_path.is_dir():
        print(f"Error: Not a directory: {train_json_path}")
        return
    
    # Auto-detect preprocessing base path
    if preprocessing_base is None:
        script_dir = Path(__file__).parent
        preprocessing_base = script_dir.parent / "preprocessing"
    else:
        preprocessing_base = Path(preprocessing_base)
    
    if not preprocessing_base.exists():
        print(f"Error: Preprocessing folder not found: {preprocessing_base}")
        return
    
    # Get original dataset_json folder for test datasets
    original_dataset_json = preprocessing_base / "dataset_json"
    if not original_dataset_json.exists():
        print(f"Warning: Original dataset_json folder not found: {original_dataset_json}")
        original_dataset_json = None
    
    # Find all JSON files in train_json folder
    json_files = sorted(train_json_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in: {train_json_path}")
        return
    
    print(f"Found {len(json_files)} JSON files in {train_json_path}")
    print(f"Creating dataset folders in: {preprocessing_base}")
    print("="*60)
    
    created_folders = []
    
    for json_file in json_files:
        # Generate folder name
        folder_name = convert_filename_to_folder_name(json_file.name)
        target_folder = preprocessing_base / folder_name
        
        # Create folder
        target_folder.mkdir(exist_ok=True)
        
        # Copy the JSON file as FaceForensics++.json
        target_json = target_folder / "FaceForensics++.json"
        shutil.copy2(json_file, target_json)
        
        print(f"✓ Created: {folder_name}/")
        print(f"  - Copied {json_file.name} -> FaceForensics++.json")
        
        # Copy test datasets if specified
        if test_datasets and original_dataset_json:
            for dataset_name in test_datasets:
                source_file = original_dataset_json / f"{dataset_name}.json"
                if source_file.exists():
                    target_file = target_folder / f"{dataset_name}.json"
                    shutil.copy2(source_file, target_file)
                    print(f"  - Copied {dataset_name}.json (test dataset)")
                else:
                    print(f"  - Warning: {dataset_name}.json not found in original dataset_json")
        
        created_folders.append(folder_name)
        print()
    
    print("="*60)
    print(f"SUCCESS: Created {len(created_folders)} dataset folders")
    print("\nCreated folders:")
    for folder_name in created_folders:
        print(f"  - {folder_name}")
    print(f"\nLocation: {preprocessing_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset folders from cartography train_json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create folders without test datasets
  python create_dataset_folders.py /path/to/train_json
  
  # Create folders with test datasets for cross-dataset evaluation
  python create_dataset_folders.py /path/to/train_json --test-datasets Celeb-DF-v2 DFDCP
  
  # Specify custom preprocessing base path
  python create_dataset_folders.py /path/to/train_json --preprocessing-base /custom/path
        """
    )
    
    parser.add_argument(
        'train_json_folder',
        type=str,
        help='Path to the train_json folder containing filtered JSON files'
    )
    
    parser.add_argument(
        '--test-datasets',
        type=str,
        nargs='+',
        default=None,
        help='List of test dataset names to copy (e.g., Celeb-DF-v2 DFDCP)'
    )
    
    parser.add_argument(
        '--preprocessing-base',
        type=str,
        default=None,
        help='Base path for preprocessing folder (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    create_dataset_folders(
        args.train_json_folder,
        test_datasets=args.test_datasets,
        preprocessing_base=args.preprocessing_base
    )


if __name__ == "__main__":
    main()
