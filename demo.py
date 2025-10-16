import os, cv2, torch, torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from src.imu_encoder import IMUEncoder
from src.data_utils import IMUPairsDataset
from src.video_encoder import VideoActionEncoder

device = "cpu"

# ✅ 固定样本路径
video_path = "/data/lxhong/mmact_data/trimmed_camera1/video/test/camera1/subject1-1/subject1/subject1-1.subject1.waving_hand.mp4"
imu_item = {
    "group": "subject1-1",
    "action": "waving_hand",
    "sensor": {
        "acc":  "/data/lxhong/mmact_data/trimmed_sensor/sensor/test/acc2_clip/camera1/person1-1/subject1/waving_hand/person1-1.subject1.waving_hand.csv",
        "gyro": "/data/lxhong/mmact_data/trimmed_sensor/sensor/test/gyro_clip/camera1/person1-1/subject1/waving_hand/person1-1.subject1.waving_hand.csv",
        "ori":  "/data/lxhong/mmact_data/trimmed_sensor/sensor/test/orientation_clip/camera1/person1-1/subject1/waving_hand/person1-1.subject1.waving_hand.csv"
    }
}

save_path = "./report/demo.jpg"
yolo_path = "/home/lxhong/mmact/utils/yolov8m.pt"
ckpt_path = "./checkpoints/imu_action.pth"


def get_imu_vec(item):
    """加载 IMU 模型并提取嵌入特征"""
    tmp_json = "./report/tmp_single.json"
    os.makedirs("./report", exist_ok=True)
    import json
    with open(tmp_json, "w") as f:
        json.dump([item], f)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = IMUEncoder(9, 128, 256).to(device).eval()
    model.load_state_dict(ckpt["encoder"])

    ds = IMUPairsDataset(tmp_json)
    x, _ = ds[0]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        feats = F.normalize(model(x), dim=-1)
    return feats


def main():
    # 1️⃣ 取视频首帧
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[ERROR] 无法加载视频帧")
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2️⃣ 检测人物
    yolo = YOLO(yolo_path)
    res = yolo(rgb, classes=[0], verbose=False, device='cpu')
    boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
    print(f"[INFO] 检测到 {len(boxes)} 人")

    # 3️⃣ 计算 IMU 向量并打印动作
    imu_vec = get_imu_vec(imu_item)

    # 🔹打印当前 IMU 样本动作标签
    print(f"[INFO] IMU 动作标签: {imu_item['action']}")

    # 4️⃣ VideoMAE 向量并匹配
    vae = VideoActionEncoder(device=device)
    sims = []
    for box in boxes:
        emb = vae.encode_person(video_path, box)

        # 临时把维度对齐
        if imu_vec.shape[1] != emb.shape[1]:
            if imu_vec.shape[1] < emb.shape[1]:
                imu_tmp = F.pad(imu_vec, (0, emb.shape[1] - imu_vec.shape[1]))
            else:
                imu_tmp = imu_vec[:, :emb.shape[1]]
        else:
            imu_tmp = imu_vec
        sims.append(float(F.cosine_similarity(imu_tmp, emb)))
    best = int(np.argmax(sims))
    vid_label_id, vid_conf = vae.predict_action(video_path, boxes[best])
    print(f"[INFO] Video 模型预测动作: {KINETICS_400_LABELS[vid_label_id]}  (置信度={vid_conf:.2f})")
    print(f"[INFO] 最相似人物ID={best}, 相似度={sims[best]:.3f}")

    # 5️⃣ 画框并保存结果
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = (0, 0, 255) if i == best else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"P{i}:{sims[i]:.2f}", (x1, max(20, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, frame)
    print(f"✅ 结果已保存到 {save_path}")


if __name__ == "__main__":
    main()