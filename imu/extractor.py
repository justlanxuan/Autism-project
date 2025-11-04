# extractor.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

class IMUExtractor:    
    def __init__(self, annotation_path: str, data_root: str):
        self.annotation_path = Path(annotation_path)
        self.data_root = Path(data_root)
        
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
            
        print(f"✓ loading {len(self.annotations)} annotations")
    
    def _resolve_path(self, path_str: str) -> Path:
        resolved = path_str.replace("${paths.data_root}", str(self.data_root))
        return Path(resolved)
    
    def extract_subject_data(self, subject_idx: int = None, subject_name: str = None) -> List[Dict]:
        if subject_name is not None:
            for i, subj in enumerate(self.annotations):
                if subj['subject'] == subject_name:
                    subject_idx = i
                    break
            if subject_idx is None:
                raise ValueError(f"cannot find: {subject_name}")
            
        subject = self.annotations[subject_idx]
        print(f"\nSubject: {subject['subject']} ({len(subject['action_vector'])} actions)")
        
        extracted_data = []
        
        for i, (action, sensor_info) in enumerate(zip(subject['action_vector'], 
                                                       subject['sensor_vector'])):
            try:
                acc_path = self._resolve_path(sensor_info['acc'])
                gyro_path = self._resolve_path(sensor_info['gyro'])
                ori_path = self._resolve_path(sensor_info['ori'])
                
                acc_df = pd.read_csv(acc_path)
                gyro_df = pd.read_csv(gyro_path)
                ori_df = pd.read_csv(ori_path)
                
                # first column is timestamp, rest are data
                acc_data = acc_df.iloc[:, 1:].values
                gyro_data = gyro_df.iloc[:, 1:].values
                ori_data = ori_df.iloc[:, 1:].values
                
                from scipy.interpolate import interp1d
                # different sensors have different sampling rates, need to align
                target_len = len(gyro_data)
                
                if len(acc_data) != target_len:
                    f = interp1d(np.arange(len(acc_data)), acc_data, axis=0, kind='linear')
                    acc_data = f(np.linspace(0, len(acc_data)-1, target_len))
                
                if len(ori_data) != target_len:
                    f = interp1d(np.arange(len(ori_data)), ori_data, axis=0, kind='linear')
                    ori_data = f(np.linspace(0, len(ori_data)-1, target_len))
                
                # 9 columns: [gyro(3), acc(3), ori(3)]
                imu = np.hstack([gyro_data, acc_data, ori_data])
                
                data_item = {
                    'subject': subject['subject'],
                    'action': action,
                    'timestamp': subject['timestamp_vector'][i],
                    'camera': subject['camera_vector'][i],
                    'imu': imu,
                    'sampling_rate': 100
                }
                
                extracted_data.append(data_item)
                print(f"  ✓ [{i+1:2d}] {action:20s} {imu.shape}")
                
            except Exception as e:
                print(f"  ✗ [{i+1:2d}] {action:20s} error: {str(e)}")
                
        return extracted_data
    
    def extract_all_subjects(self) -> Dict[str, List[Dict]]:
        all_data = {}
        
        for i, subject in enumerate(self.annotations):
            subject_name = subject['subject']
            try:
                data = self.extract_subject_data(subject_idx=i)
                all_data[subject_name] = data
                print(f"✓ {subject_name}: {len(data)} actions")
            except Exception as e:
                print(f"✗ {subject_name} fail: {str(e)}")
                
        return all_data
    
    def save_extracted_data(self, data_list: List[Dict], output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for item in data_list:
            subject = item['subject']
            action = item['action']
            filename = f"{subject}_{action}.txt"
            filepath = output_dir / filename
            
            np.savetxt(filepath, item['imu'], fmt='%.6f')
            
        print(f"✓ saved {len(data_list)} files to {output_dir}")