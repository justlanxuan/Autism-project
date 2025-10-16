import json
import os
import numpy as np
import pandas as pd
from src.data_utils import IMUPairsDataset


def inspect_alignment(json_path, save_path="./report/test_pairs.json"):
    dataset = IMUPairsDataset(json_path)
    results = []

    print(f"[INFO] 检查 {len(dataset)} 个样本...")

    for i, item in enumerate(dataset.items[:2]):  # 这里只取前 2 个样本
        acc_p = item["sensor"]["acc"]
        gyro_p = item["sensor"]["gyro"]
        ori_p = item["sensor"]["ori"]
        group = item["group"]
        action = item["action"]

        # --- 读取原始信号 ---
        def read_csv_len(p):
            if not os.path.exists(p):
                return 0
            try:
                df = pd.read_csv(p)
                return len(df)
            except Exception:
                return 0

        acc_len = read_csv_len(acc_p)
        gyro_len = read_csv_len(gyro_p)

        # --- 运行我们的对齐流程 ---
        feat = dataset._align_modalities(acc_p, gyro_p, ori_p)
        aligned_frames = feat.shape[0]

        results.append({
            "group": group,
            "action": action,
            "frames_acc": int(acc_len),
            "frames_gyro": int(gyro_len),
            "frames_after_align": int(aligned_frames)
        })

        print(f"[{i}] {group:<12} {action:<12} acc={acc_len:<5} gyro={gyro_len:<5} after={aligned_frames:<5}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ 已保存结果到: {save_path}")


if __name__ == "__main__":
    annotation_json = "/home/lxhong/mmact/report/annotation_pairs.json"
    inspect_alignment(annotation_json)