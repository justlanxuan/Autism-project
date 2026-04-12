"""
WHAM 3D 人体姿态估计器适配器
直接调用 WHAM 完整处理流程，得到 3D 人体网格和参数
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List


class WHAM3DConfig:
    """WHAM 3D 估计器配置"""
    def __init__(self, config_dict):
        self.repo_root = config_dict.get('repo_root', '/home/fzliang/origin/WHAM')
        self.checkpoint_file = config_dict.get('checkpoint_file', None)
        self.device = config_dict.get('device', 'cuda:0')
        self.run_global = config_dict.get('run_global', True)
        self.output_dir = config_dict.get('output_dir', './wham_outputs')
        

class WHAM3DEstimator:
    """
    WHAM 3D 人体估计器
    接收视频文件路径，输出 3D 人体网格、SMPL 参数等
    """
    
    def __init__(self, config_dict: Dict):
        self.config = WHAM3DConfig(config_dict)
        self._wham_api = None
        self._initialized = False
        
    def _load_wham(self):
        """懒加载 WHAM 模型"""
        if self._initialized:
            return
        
        # 懒加载依赖
        import torch
        import joblib
        import numpy as np
            
        # 添加 WHAM 路径到 sys.path
        wham_path = self.config.repo_root
        if wham_path not in sys.path:
            sys.path.insert(0, wham_path)
        
        try:
            from wham_api import WHAM_API
            self._wham_api = WHAM_API()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to load WHAM from {wham_path}: {str(e)}")
    
    def reset(self):
        """重置估计器状态"""
        pass
    
    def process_video(self, video_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        处理视频文件，获得 3D 人体结果
        
        Args:
            video_path: 输入视频文件路径
            output_dir: 输出结果保存目录
            
        Returns:
            Dict: 3D 人体结果字典，格式为 {person_id: {...3D人体数据...}}
                  每个 person 包含:
                  - poses_body: (T, 63) SMPL 身体参数
                  - poses_root_cam: (T, 3) 根节点旋转（相机坐标）
                  - poses_root_world: (T, 3) 根节点旋转（世界坐标）
                  - betas: (10,) 身体形状参数
                  - trans_world: (T, 3) 3D 位置（世界坐标）
                  - verts_cam: (T, 6890, 3) 3D 网格顶点（相机坐标）
                  - frame_id: (T,) 帧索引
        """
        # 加载 WHAM 模型
        self._load_wham()
        
        # 确定输出目录
        if output_dir is None:
            output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查视频文件
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"[WHAM3D] Processing video: {video_path}")
        print(f"[WHAM3D] Output directory: {output_dir}")
        print(f"[WHAM3D] Device: {self.config.device}")
        print(f"[WHAM3D] Run global: {self.config.run_global}")
        
        try:
            # 调用 WHAM API 进行完整处理
            results, tracking_results, slam_results = self._wham_api(
                video_path,
                output_dir=output_dir,
                run_global=self.config.run_global,
                visualize=False
            )
            
            print(f"[WHAM3D] Processing completed!")
            print(f"[WHAM3D] Detected {len(results)} persons")
            
            return {
                'results_3d': results,
                'tracking_results': tracking_results,
                'slam_results': slam_results
            }
            
        except Exception as e:
            raise RuntimeError(f"WHAM processing failed: {str(e)}")
    
    def __call__(self, video_path: str, output_dir: Optional[str] = None) -> Dict:
        """调用接口"""
        return self.process_video(video_path, output_dir)


def build_wham_3d_estimator(config_dict: Dict) -> WHAM3DEstimator:
    """构建 WHAM 3D 估计器"""
    return WHAM3DEstimator(config_dict)
