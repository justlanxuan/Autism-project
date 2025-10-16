import os
import re
import pandas as pd
import numpy as np
import torch


class MMActLoader:
    """
    MMAct 数据加载：
    - annotation 与 sensor 通过同名文件夹 (subjectX_Y ↔ personX_Y) 对应；
    - annotation 行格式:
         2021/11/19 11:16:19.121-2021/11/19 11:16:22.308-standing-1
    - sensor CSV: 时间戳,x,y,z
    """

    def __init__(self, root, camera="camera1", sensor_type="acc2_clip"):
        self.root = root
        self.ann_root = os.path.join(root, "annotation")
        self.sensor_root = os.path.join(
            root, "trimmed_sensor", "sensor", "test", sensor_type, camera
        )

    # ------------------------------------------------------------------
    def load_annotation(self):
        """
        读取 annotation 文件，解析出 (group, subject, action, start, end)
        """
        metas = []
        line_pattern = re.compile(
            r"(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)-"
            r"(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)-([A-Za-z_]+)"
        )

        for group in sorted(os.listdir(self.ann_root)):
            grp_path = os.path.join(self.ann_root, group)
            if not os.path.isdir(grp_path):
                continue

            for fname in sorted(os.listdir(grp_path)):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(grp_path, fname)
                parts = fname.split(".")
                if len(parts) < 3:
                    continue
                file_subject = parts[1]

                with open(fpath, "r") as fin:
                    for raw in fin:
                        line = raw.strip()
                        if not line:
                            continue
                        m = line_pattern.search(line)
                        if not m:
                            continue
                        start, end, action = m.groups()
                        # 转换时间格式
                        start_ts = start.replace("/", "-")
                        end_ts = end.replace("/", "-")
                        metas.append(
                            dict(
                                group=group,
                                subject=file_subject,
                                action=action,
                                ann_start=start_ts,
                                ann_end=end_ts,
                                ann_file=fpath,
                            )
                        )
        return pd.DataFrame(metas)

    # ------------------------------------------------------------------

    def load_sensor(self, group, subject, action):
        """
        从 sensor 目录读取匹配的 IMU CSV。
        优先当前 camera，其它 camera (2→3→4) 作为后备。
        返回 (feature_tensor, csv_path)
        """
        # 所有候选 camera
        camera_list = ["camera1", "camera2", "camera3", "camera4"]

        # 构造 base path: e.g. "trimmed_sensor/sensor/test/acc2_clip"
        base_root = os.path.join(self.root, "trimmed_sensor", "sensor", "test", "acc2_clip")

        person_dir = group.replace("subject", "person")

        for cam in camera_list:
            cam_path = os.path.join(base_root, cam)
            person_path = os.path.join(cam_path, person_dir)
            subj_path = os.path.join(person_path, subject)
            act_path = os.path.join(subj_path, action)

            if not os.path.isdir(act_path):
                continue

            # 找到目录 → 搜索第一个 csv
            for fname in sorted(os.listdir(act_path)):
                if not fname.endswith(".csv"):
                    continue
                csv_path = os.path.join(act_path, fname)
                try:
                    df = pd.read_csv(csv_path)
                    vals = df.iloc[:, 1:].to_numpy(float)
                    if vals.size == 0:
                        return torch.zeros(6), csv_path
                    feat = np.r_[vals.mean(0), vals.std(0)]
                    return torch.tensor(feat, dtype=torch.float32), csv_path
                except Exception as e:
                    print(f"[WARN] Failed to read {csv_path}: {e}")
                    return torch.zeros(6), csv_path

        # 所有 camera 都没找到
        return torch.zeros(6), None

    # ------------------------------------------------------------------
    def alignment_check(self, threshold_sec=300):
        """
        比较 annotation 的 start / sensor CSV 的首帧时间。
        """
        meta = self.load_annotation()
        results = []

        for _, row in meta.iterrows():
            group = row["group"]
            subject = row["subject"]
            action = row["action"]

            try:
                ann_start = pd.to_datetime(row["ann_start"])
                ann_end = pd.to_datetime(row["ann_end"])
            except Exception:
                continue

            _, csv_path = self.load_sensor(group, subject, action)

            imu_start = imu_end = None
            imu_ok = False
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    try:
                        imu_start = pd.to_datetime(df.iloc[0, 0])
                        imu_end = pd.to_datetime(df.iloc[-1, 0])
                        diff = abs((imu_start - ann_start).total_seconds())
                        imu_ok = diff < threshold_sec
                    except Exception:
                        pass

            results.append(
                dict(
                    group=group,
                    subject=subject,
                    action=action,
                    ann_start=ann_start,
                    ann_end=ann_end,
                    imu_start=imu_start,
                    imu_end=imu_end,
                    imu_ok=imu_ok,
                    csv_path=csv_path,
                )
            )

        return pd.DataFrame(results)