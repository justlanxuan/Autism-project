import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class IMUPairsDataset(Dataset):

    def __init__(self, json_path, max_len=300):
        super().__init__()
        self.items = json.load(open(json_path))
        self.max_len = max_len
        self.actions = sorted(list(set(item["action"] for item in self.items)))
        self.act2id = {a: i for i, a in enumerate(self.actions)}

    def __len__(self):
        return len(self.items)

    # ------------------------------------------------------------------
    def _read_csv(self, path):
        if not path or not os.path.exists(path):
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float64)

        df = pd.read_csv(path)
        if df.shape[0] == 0:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float64)

        try:
            times = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            times = times.astype('int64') / 1e9 
            vals = df.iloc[:, 1:4].to_numpy(np.float32)
            mask = ~np.isnan(times)
            times = times[mask]
            vals = vals[mask]
        except Exception as e:
            print(f"[WARN] Reading {path} Fail: {e}")
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float64)
        return vals, times

    # ------------------------------------------------------------------
    def _align_modalities(self, acc_path, gyro_path, ori_path):
        acc, acc_t = self._read_csv(acc_path)
        gyro, gyro_t = self._read_csv(gyro_path)
        ori, ori_t = self._read_csv(ori_path)

        if len(acc_t) == 0:
            return np.zeros((self.max_len, 9), np.float32)

        gyro_aligned = np.zeros_like(acc)
        ori_aligned = np.zeros_like(acc)

        if len(gyro_t) > 0:
            gi = 0
            for i, t in enumerate(acc_t):
                while gi + 1 < len(gyro_t) and abs(gyro_t[gi + 1] - t) < abs(gyro_t[gi] - t):
                    gi += 1
                gyro_aligned[i] = gyro[gi]
        else:
            gyro_aligned[:] = 0

        if len(ori_t) > 0:
            oi = 0
            for i, t in enumerate(acc_t):
                while oi + 1 < len(ori_t) and abs(ori_t[oi + 1] - t) < abs(ori_t[oi] - t):
                    oi += 1
                ori_aligned[i] = ori[oi]
        else:
            ori_aligned[:] = 0

        feat = np.concatenate([acc, gyro_aligned, ori_aligned], axis=1)
        return feat

    # ------------------------------------------------------------------
    def _pad_crop(self, arr):
        T = len(arr)
        if T == 0:
            return np.zeros((self.max_len, 9), np.float32)
        if T < self.max_len:
            pad = np.repeat(arr[-1:], self.max_len - T, axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif T > self.max_len:
            arr = arr[:self.max_len]
        return arr

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        item = self.items[idx]
        acc_p = item["sensor"]["acc"]
        gyro_p = item["sensor"]["gyro"]
        ori_p = item["sensor"]["ori"]

        feat = self._align_modalities(acc_p, gyro_p, ori_p)
        feat = self._pad_crop(feat)
        lbl = self.act2id[item["action"]]
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)