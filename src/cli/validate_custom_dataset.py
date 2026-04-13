#!/usr/bin/env python3
"""
Quick validation of Custom 4-Fold dataset structure and integrity.
"""

import sys
from pathlib import Path
import csv
import numpy as np

PROJECT_ROOT = Path("/home/fzliang/Autism-project").resolve()
PROCESSED_DIR = PROJECT_ROOT / "data/processed/custom_4fold"

def validate_fold(fold_idx: int) -> bool:
    """Validate a single fold's structure and data."""
    fold_dir = PROCESSED_DIR / f"folds/fold_{fold_idx}"
    
    print(f"\n[Fold {fold_idx}]")
    print("=" * 60)
    
    if not fold_dir.exists():
        print(f"❌ Fold directory not found: {fold_dir}")
        return False
    
    print(f"✅ Fold directory exists: {fold_dir}")
    
    # Check CSVs
    csvs = {
        'train': fold_dir / 'windows_train.csv',
        'val': fold_dir / 'windows_val.csv',
        'test': fold_dir / 'windows_test.csv',
    }
    
    stats = {}
    for name, csv_path in csvs.items():
        if not csv_path.exists():
            print(f"❌ Missing CSV: {csv_path}")
            return False
        
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        stats[name] = len(rows)
        print(f"✅ {name:5s} CSV: {len(rows):4d} windows")
        
        # Validate columns
        if rows:
            required_cols = {'npz_path', 'window_start', 'window_end', 'person_label', 'imu_label'}
            actual_cols = set(rows[0].keys())
            if not required_cols.issubset(actual_cols):
                print(f"   ⚠️  Missing columns: {required_cols - actual_cols}")
            
            # Check person/imu labels
            persons = set(r.get('person_label', '?') for r in rows)
            imus = set(r.get('imu_label', '?') for r in rows)
            print(f"   📊 Persons: {sorted(persons)}, IMUs: {sorted(imus)}")
    
    # Spot check some npz files
    # NOTE: npz files are stored in PROCESSED_DIR/sequences/, shared across all folds
    print("\n📄 Spot-checking npz files (shared across all folds)...")
    with open(csvs['train']) as f:
        first_rows = list(csv.DictReader(f))[:3]
    
    for i, row in enumerate(first_rows):
        npz_rel = row['npz_path']
        # Folds reference npz files in parent PROCESSED_DIR
        npz_path = PROCESSED_DIR / npz_rel
        if not npz_path.exists():
            print(f"  ❌ Row {i}: npz not found: {npz_path}")
            return False
        
        try:
            data = np.load(npz_path)
            imu_shape = data['imu'].shape if 'imu' in data else None
            skel_shape = data['skeleton'].shape if 'skeleton' in data else None
            print(f"  ✅ Row {i}: IMU {imu_shape}, Skeleton {skel_shape}")
        except Exception as e:
            print(f"  ❌ Row {i}: Error reading npz: {e}")
            return False
    
    print(f"\n✅ Fold {fold_idx} validation PASSED")
    print(f"   Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
    print(f"   Data root should be: {PROCESSED_DIR}")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("Custom 4-Fold Dataset Validation")
    print("=" * 60)
    
    if not PROCESSED_DIR.exists():
        print(f"❌ Processed directory not found: {PROCESSED_DIR}")
        sys.exit(1)
    
    print(f"✅ Processed directory: {PROCESSED_DIR}")
    
    all_valid = True
    for fold in range(1, 5):
        if not validate_fold(fold):
            all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ ALL FOLDS VALIDATED SUCCESSFULLY")
        print("\nDataset Structure:")
        print(f"  Root: {PROCESSED_DIR}")
        print(f"  Shared sequences: {PROCESSED_DIR}/sequences/")
        print(f"  Folds: {PROCESSED_DIR}/folds/fold_{{1,2,3,4}}/")
        print("\n⚠️  IMPORTANT for training: Use --data_root pointing to the parent directory:")
        print(f"  --data_root {PROCESSED_DIR}")
        print("\nYou can now run training:")
        print("  bash /home/fzliang/Autism-project/scripts/run_custom_4fold_pipeline.sh")
    else:
        print("❌ VALIDATION FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
