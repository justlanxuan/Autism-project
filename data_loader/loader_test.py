import os
import json
import pandas as pd
import numpy as np
import torch
from loader import MMActLoader 

ROOT = "/data/lxhong/mmact_data"
REPORT_DIR = os.path.expanduser("~/mmact/report")
os.makedirs(REPORT_DIR, exist_ok=True)

loader = MMActLoader(ROOT)
df = loader.alignment_check(threshold_sec=300)

ok_ratio = 100 * df["imu_ok"].mean()
print(f"\nAlignment Check: IMU aligned {df['imu_ok'].sum()}/{len(df)} ({ok_ratio:.1f}%)")

csv_path = os.path.join(REPORT_DIR, "alignment_report.csv")
df.to_csv(csv_path, index=False)
print(f"alignment_report.csv saved to {csv_path}")

missing_df = df[~df["imu_ok"]].copy()
for col in ["ann_start", "ann_end", "imu_start", "imu_end"]:
    if col in missing_df.columns:
        missing_df[col] = missing_df[col].astype(str)
missing_dict = missing_df.to_dict(orient="records")

missing_path = os.path.join(REPORT_DIR, "missing_report.json")
with open(missing_path, "w", encoding="utf-8") as fout:
    json.dump(missing_dict, fout, ensure_ascii=False, indent=2)
print(f"missing_report.json saved ({len(missing_dict)} not aligned)")

aligned_df = df[df["imu_ok"]].copy()
annotations = []

for _, row in aligned_df.iterrows():
    group = row["group"]
    subject = row["subject"]
    action = row["action"]

    feat_tensor, csv_path = loader.load_sensor(group, subject, action)
    if csv_path is None or not os.path.exists(csv_path):
        continue

    path_parts = csv_path.split(os.sep)
    camera = next((p for p in path_parts if p.startswith("camera")), "camera1")

    gyro_path = csv_path.replace("acc2_clip", "gyro_clip")
    ori_path = csv_path.replace("acc2_clip", "orientation_clip")

    sensor_paths = {
        "acc": csv_path,
        "gyro": gyro_path,
        "ori": ori_path,
    }

    video_path = os.path.join(
        ROOT, "trimmed_camera1/video/test",
        camera, group, subject,
        f"{group}.{subject}.{action}.mp4"
    )

    record = {
        "group": group,
        "subject": subject,
        "action": action,
        "camera": camera,
        "video": video_path,
        "sensor": sensor_paths,
    }
    annotations.append(record)

print(f"Match {len(annotations)} Samples")

anno_json = os.path.join(REPORT_DIR, "annotation_pairs.json")
with open(anno_json, "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print(f"annotation_pairs.json saved to {anno_json}")

cam_counts = pd.Series([a["camera"] for a in annotations]).value_counts()
print("\n===== Camera =====")
print(cam_counts)
print("🔹 Done.")