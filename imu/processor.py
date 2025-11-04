"""
IMU批量处理器 - 使用IMUTracker批量处理数据
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict
import pickle
from tqdm import tqdm

class IMUProcessor:    
    def __init__(self, sampling_rate: int = 100, output_dir: str = "results"):
        self.sampling_rate = sampling_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def process_extracted_data(self, 
                              data_item: Dict,
                              init_range: tuple = (0, 30), # init range
                              zupt_threshold: float = 0.2,
                              filter_accel: bool = False) -> Dict:
        from main import IMUTracker
        
        data = data_item['imu']
        name = f"{data_item['subject']}_{data_item['action']}"

        tracker = IMUTracker(sampling=self.sampling_rate)

        init_start, init_end = init_range
        init_list = tracker.initialize(data[init_start:init_end])
        a_nav, orix, oriy, oriz = tracker.attitudeTrack(data[init_end:], init_list)
        a_nav_filtered = tracker.removeAccErr(a_nav, filter=filter_accel)
        velocity = tracker.zupt(a_nav_filtered, threshold=zupt_threshold)

        position = tracker.positionTrack(a_nav_filtered, velocity)
        
        result = {
            'name': name,
            'subject': data_item['subject'],
            'action': data_item['action'],
            'timestamp': data_item['timestamp'],
            'position': position,
            'velocity': velocity,
            'acceleration': a_nav_filtered,
            'orientation_x': orix,
            'orientation_y': oriy,
            'orientation_z': oriz,
            'sampling_rate': self.sampling_rate
        }
        
        save_data = self.output_dir / result['subject'] / result['action']
        save_img = self.output_dir / result['subject']
        save_data.mkdir(parents=True, exist_ok=True)
        save_img.mkdir(parents=True, exist_ok=True)
        self._save_result(result, save_data)
        self._save_visualization(result, save_img)
        
        print(f"✓ 完成")
        return result
    
    def process_batch(self, data_list: List[Dict], **kwargs) -> List[Dict]:
        self.results = []
        for i, data_item in enumerate(data_list, 1):
            print(f"\n[{i}/{len(data_list)}]")
            try:
                result = self.process_extracted_data(data_item, **kwargs)
                self.results.append(result)
            except Exception as e:
                print(f"fails to process: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def _save_result(self, result: Dict, save_dir: Path):

        np.savetxt(save_dir / 'position.txt', result['position'])
        np.savetxt(save_dir / 'velocity.txt', result['velocity'])
        np.savetxt(save_dir / 'acceleration.txt', result['acceleration'])

        with open(save_dir / 'result.pkl', 'wb') as f:
            pickle.dump(result, f)
    
    def _save_visualization(self, result: Dict, save_dir: Path):
        from plotlib import plot3
        
        name = result['name']
        # plot3
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'{name}', fontsize=16)
        
        ax1 = [plt.subplot(3, 3, 1), plt.subplot(3, 3, 2), plt.subplot(3, 3, 3)]
        ax1[0].set_title('Acceleration')
        plot3([result['acceleration']], 
              ax=ax1, 
              labels=[['Acc X', 'Acc Y', 'Acc Z']], 
              show_legend=True,
              show=False)
        
        ax2 = [plt.subplot(3, 3, 4), plt.subplot(3, 3, 5), plt.subplot(3, 3, 6)]
        ax2[0].set_title('Velocity')
        plot3([result['velocity']], 
              ax=ax2,
              labels=[['Vel X', 'Vel Y', 'Vel Z']], 
              show_legend=True,
              show=False)
        
        orientation = np.column_stack([
            result['orientation_x'],
            result['orientation_y'],
            result['orientation_z']
        ])
        ax3 = [plt.subplot(3, 3, 7), plt.subplot(3, 3, 8), plt.subplot(3, 3, 9)]
        ax3[0].set_title('Orientation')
        plot3([orientation], 
              ax=ax3,
              labels=[['Ori X', 'Ori Y', 'Ori Z']], 
              show_legend=True,
              show=False)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{name} - Plot3', dpi=150
                    #, bbox_inches='tight'
                    )
        plt.close(fig)
        
        # plot 3d
        from plotlib import plot3D

        fig2 = plt.figure(figsize=(10, 8))
        ax = fig2.add_subplot(111, projection='3d')

        pos = result['position']
        data = [[pos, name]]
        plot3D(data, ax=ax)

        ax.plot([pos[-1,0]], [pos[-1,1]], [pos[-1,2]], 'go', 
                markersize=10, label='End')
        ax.set_title(f'{name} - 3D Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.grid(True)

        plt.savefig(save_dir / f'{name} - 3D Trajectory', dpi=150, 
                    #bbox_inches='tight'
                    )
        plt.close(fig2)

    def generate_report(self):
        """生成处理报告"""
        if not self.results:
            print("⚠️  没有可用的处理结果")
            return
        
        print("\n" + "="*80)
        print(f"📊 处理报告 - 共 {len(self.results)} 个样本")
        print("="*80)
        
        # 表头
        print(f"\n{'序号':<4} {'受试者':<12} {'动作':<15} {'路程(m)':<10} "
              f"{'最大速度(m/s)':<15} {'最终位置(m)'}")
        print("─" * 80)
        
        for i, result in enumerate(self.results, 1):
            pos = result['position']
            vel = result['velocity']
            
            # 计算总路程
            total_dist = np.sum(np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1)))
            
            # 最大速度
            speed = np.sqrt(np.sum(vel**2, axis=1))
            max_speed = np.max(speed)
            
            # 最终位置
            final_pos = f"[{pos[-1,0]:.2f}, {pos[-1,1]:.2f}, {pos[-1,2]:.2f}]"
            
            print(f"{i:<4} {result['subject']:<12} {result['action']:<15} "
                  f"{total_dist:<10.3f} {max_speed:<15.3f} {final_pos}")
        
        
        # 按动作统计
        print("📈 按动作类型统计:")
        print("─" * 60)
        
        actions_stats = {}
        for result in self.results:
            action = result['action']
            pos = result['position']
            dist = np.sum(np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1)))
            
            if action not in actions_stats:
                actions_stats[action] = []
            actions_stats[action].append(dist)
        
        for action, dists in actions_stats.items():
            print(f"  {action:15s}: n={len(dists):2d}, "
                  f"平均路程={np.mean(dists):.3f}±{np.std(dists):.3f} m")
    