import json
import pandas as pd
import numpy as np
import cv2
import os
from collections import defaultdict


def load_sample(sample_dict, num_frames=16):
    """
    Args:
        sample_dict: annotation_pairs
        num_frames: 视频采样帧数
    
    Returns:
        dict 或 None
    """
    # 加载视频
    video_path = sample_dict['video']
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total == 0 or fps == 0:
        cap.release()
        return None
    
    frames = []
    times = []
    for idx in np.linspace(0, total - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        times.append(idx / fps)
    
    cap.release()
    
    # 加载IMU
    sensors = {}
    for name in ['acc', 'gyro', 'ori']:
        csv_path = sample_dict['sensor'][name]
        if not os.path.exists(csv_path):
            return None
        
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                return None
            
            ts = pd.to_datetime(df.iloc[:, 0])
            data = df.iloc[:, 1:4].to_numpy()
            rel_time = (ts - ts[0]).total_seconds().to_numpy()
            
            aligned = [data[np.argmin(np.abs(rel_time - t))] for t in times]
            sensors[name] = np.array(aligned)
        except:
            return None
    
    return {
        'video': np.array(frames),
        'sensor': sensors,
        'action': sample_dict['action'],
        'subject': sample_dict['subject']
    }


def load_continuous_sequence(annotations, alignment_report_path, subject_id, num_frames_per_action=16):
    """
    模式2: 加载一个subject的连续动作序列
    
    Args:
        annotations: annotation_pairs.json 的内容
        alignment_report_path: alignment_report.csv 路径
        subject_id: 目标subject (例如 'subject1')
        num_frames_per_action: 每个动作采样的帧数
    
    Returns:
        {
            'video': np.array (T_total, H, W, 3),
            'sensor': {
                'acc': np.array (T_total, 3),
                'gyro': np.array (T_total, 3),
                'ori': np.array (T_total, 3)
            },
            'actions': list of str,  # 动作序列
            'action_boundaries': list of int,  # 每个动作的起始帧索引
            'subject': str
        }
        或 None
    """
    # 读取 alignment report
    alignment_df = pd.read_csv(alignment_report_path)
    
    # 过滤出该 subject 的所有动作（按时间排序）
    subject_actions = alignment_df[alignment_df['subject'] == subject_id].copy()
    if len(subject_actions) == 0:
        return None
    
    # 按开始时间排序
    subject_actions['ann_start'] = pd.to_datetime(subject_actions['ann_start'])
    subject_actions = subject_actions.sort_values('ann_start')
    
    # 收集该 subject 的所有样本
    all_videos = []
    all_sensors = {'acc': [], 'gyro': [], 'ori': []}
    action_sequence = []
    action_boundaries = [0]
    
    total_frames = 0
    
    for _, row in subject_actions.iterrows():
        action = row['action']
        
        # 在 annotations 中找到对应的样本
        matching_samples = [
            ann for ann in annotations 
            if ann['subject'] == subject_id and ann['action'] == action
        ]
        
        if len(matching_samples) == 0:
            continue
        
        # 加载第一个匹配的样本
        sample_dict = matching_samples[0]
        sample = load_sample(sample_dict, num_frames=num_frames_per_action)
        
        if sample is None:
            continue
        
        # 累积数据
        all_videos.append(sample['video'])
        all_sensors['acc'].append(sample['sensor']['acc'])
        all_sensors['gyro'].append(sample['sensor']['gyro'])
        all_sensors['ori'].append(sample['sensor']['ori'])
        
        action_sequence.append(action)
        total_frames += num_frames_per_action
        action_boundaries.append(total_frames)
    
    if len(all_videos) == 0:
        return None
    
    # 拼接所有片段
    return {
        'video': np.concatenate(all_videos, axis=0),
        'sensor': {
            'acc': np.concatenate(all_sensors['acc'], axis=0),
            'gyro': np.concatenate(all_sensors['gyro'], axis=0),
            'ori': np.concatenate(all_sensors['ori'], axis=0)
        },
        'actions': action_sequence,
        'action_boundaries': action_boundaries[:-1],  # 去掉最后一个
        'subject': subject_id
    }


def load_dataset(annotation_path, num_frames=16):
    """
    模式1批量加载: 每个动作单独加载
    """
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    dataset = []
    for sample_dict in annotations:
        data = load_sample(sample_dict, num_frames)
        if data is not None:
            dataset.append(data)
    
    return dataset


def load_continuous_dataset(annotation_path, alignment_report_path, num_frames_per_action=16):
    """
    模式2批量加载: 每个subject的连续动作序列
    
    Returns:
        list of continuous sequences
    """
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    alignment_df = pd.read_csv(alignment_report_path)
    
    # 获取所有unique subjects
    subjects = alignment_df['subject'].unique()
    
    dataset = []
    for subject_id in subjects:
        seq = load_continuous_sequence(
            annotations, 
            alignment_report_path, 
            subject_id, 
            num_frames_per_action
        )
        if seq is not None:
            dataset.append(seq)
    
    return dataset