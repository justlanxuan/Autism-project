import os, cv2, random, numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import mediapipe as mp

# ========= 初始化 =========
def init_yolo():
    return YOLO("./checkpoints/yolov8m.pt")

def draw_point(frame, point, color):
    if np.any(point[:2]) and point[2] > 0.1:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 4, color, -1)

def detect_person_yolo(yolo_model, frame, conf=0.5):
    results = yolo_model.predict(frame, conf=conf, verbose=False)
    boxes = []
    if len(results) > 0:
        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = box
            if int(cls) == 0 and score >= conf:
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


# ========= 各种算法的右腕提取 =========

def get_wrists_mediapipe(frame, mp_pose, person_boxes):
    wrists = []
    h_img, w_img = frame.shape[:2]
    with mp_pose.Pose(static_image_mode=True) as pose:
        for (x1, y1, x2, y2) in person_boxes:
            crop = frame[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
            if crop.size == 0:
                wrists.append(np.zeros(3))
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                wrist_x = x1 + lm.x * (x2 - x1)
                wrist_y = y1 + lm.y * (y2 - y1)
                wrists.append(np.array([wrist_x, wrist_y, lm.visibility]))
            else:
                wrists.append(np.zeros(3))
    return wrists


def get_wrists_mmpose(frame, model, person_boxes):
    from mmpose.apis import inference_topdown_pose_model
    wrists = []
    persons = [{'bbox': box} for box in person_boxes]
    pose_results = inference_topdown_pose_model(model, frame, persons, bbox_thr=None, format='xyxy')
    for res in pose_results:
        if 'keypoints' in res:
            x, y, c = res['keypoints'][4]
            wrists.append(np.array([x, y, c]))
    return wrists


def get_wrists_openpose(frame, opWrapper):
    from openpose import pyopenpose as op
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    wrists = []
    if datum.poseKeypoints is not None:
        for person in datum.poseKeypoints:
            x, y, c = person[4]
            wrists.append(np.array([x, y, c]))
    return wrists


# ========= 主函数 =========
def preview_video(video_path, methods=("mediapipe",), save_root="report/preview_frames"):
    os.makedirs(save_root, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(random.sample(range(total), min(10, total)))

    # YOLO 初始化一次
    yolo_model = init_yolo()

    # Mediapipe 初始化
    mp_pose = mp.solutions.pose

    # 若含 MMPose，则初始化
    mmpose_model = None
    if "mmpose" in methods:
        from mmpose.apis import init_pose_model
        cfg = "configs/hrnet_w32_coco_256x192.py"
        ckpt = "checkpoints/hrnet_w32_coco_256x192.pth"
        mmpose_model = init_pose_model(cfg, ckpt, device="cuda")

    # 若含 OpenPose，则初始化
    openpose_model = None
    if "openpose" in methods:
        from openpose import pyopenpose as op
        openpose_model = op.WrapperPython()
        openpose_model.configure({"model_folder": "./models/"})
        openpose_model.start()

    # ====== 抽帧测试 ======
    for method in methods:
        save_dir = os.path.join(save_root, method)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n🎯 Testing {method} → {save_dir}")

        for i, idx in enumerate(tqdm(indices, desc=f"{method} frames")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue

            boxes = detect_person_yolo(yolo_model, frame)
            if method == "mediapipe":
                wrists = get_wrists_mediapipe(frame, mp_pose, boxes)
                color = (0, 255, 0)  # 绿
            elif method == "mmpose":
                wrists = get_wrists_mmpose(frame, mmpose_model, boxes)
                color = (255, 0, 0)  # 蓝
            elif method == "openpose":
                wrists = get_wrists_openpose(frame, openpose_model)
                color = (255, 255, 255)  # 白
            else:
                continue

            # 绘制方框和点
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for w in wrists:
                draw_point(frame, w, color)

            save_path = os.path.join(save_dir, f"{os.path.basename(video_path)}_frame{i+1}.jpg")
            cv2.imwrite(save_path, frame)

    cap.release()
    print(f"\n✅ 所有方法测试帧已保存到: {os.path.abspath(save_root)}")


if __name__ == "__main__":
    test_video = "report/results_skeletons/subject1-1.subject1.walking_skeleton.mp4"
    # 可选项：('mediapipe',), ('mmpose',), ('openpose',)
    preview_video(test_video, methods=("mediapipe","mmpose"))