import json, os, cv2, numpy as np
from tqdm import tqdm
import yaml

##########################
# init
##########################

def init_mediapipe():
    import mediapipe as mp
    return mp.solutions.pose

def init_openpose():
    from openpose import pyopenpose as op
    opWrapper = op.WrapperPython()
    opWrapper.configure({"model_folder": "./models/"})
    opWrapper.start()
    return opWrapper

def init_mmpose():
    from mmpose.apis import init_pose_model
    cfg = "configs/hrnet_w32_coco_256x192.py"
    ckpt = "checkpoints/hrnet_w32_coco_256x192.pth"
    model = init_pose_model(cfg, ckpt, device="cuda")
    return model

def init_yolo():
    # ✅ 初始化 YOLO，用于人体检测
    from ultralytics import YOLO
    model = YOLO("./checkpoints/yolov8m.pt")  # 你已有的权重路径
    return model

##########################
# extract
##########################
def detect_person_yolo(yolo_model, frame, conf=0.5):
    """使用 YOLO 检测人体，返回人框 [x1,y1,x2,y2] 列表"""
    results = yolo_model.predict(frame, conf=conf, verbose=False)
    boxes = []
    if len(results) > 0:
        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = box
            if int(cls) == 0 and score >= conf:  # class 0 = 人
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

def get_wrists_mediapipe(frame, mp_pose, person_boxes):
    """对每个人裁剪区域执行 Mediapipe"""
    wrists = []
    h_img, w_img = frame.shape[:2]
    with mp_pose.Pose(static_image_mode=True) as pose:
        for (x1,y1,x2,y2) in person_boxes:
            crop = frame[max(0,y1):min(h_img,y2), max(0,x1):min(w_img,x2)]
            if crop.size == 0:
                wrists.append(np.zeros(3))
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                # 将裁剪坐标映射回全图
                wrist_x = x1 + lm.x * (x2 - x1)
                wrist_y = y1 + lm.y * (y2 - y1)
                wrists.append(np.array([wrist_x, wrist_y, lm.visibility]))
            else:
                wrists.append(np.zeros(3))
    return wrists

def get_wrists_mmpose(frame, model, person_boxes):
    """逐人执行 MMPose，返回每人右手腕"""
    from mmpose.apis import inference_topdown_pose_model

    person_results = [{'bbox': box} for box in person_boxes]
    pose_results = inference_topdown_pose_model(model, frame, person_results, bbox_thr=None, format='xyxy')
    wrists=[]
    for res in pose_results:
        if 'keypoints' in res:
            x,y,c = res['keypoints'][4]  # 右手腕
            wrists.append(np.array([x,y,c]))
    return wrists

def get_wrist_openpose(frame, opWrapper):
    from openpose import pyopenpose as op
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    if datum.poseKeypoints is None:
        return []
    wrists=[]
    for person in datum.poseKeypoints:
        x,y,c = person[4]
        wrists.append(np.array([x,y,c]))
    return wrists

##########################
# vis
##########################

def draw_point(frame, point, color):
    if np.any(point[:2]) and point[2]>0.1:
        x,y=int(point[0]),int(point[1])
        cv2.circle(frame,(x,y),4,color,-1)

##########################
# process
##########################

def process_video(video_path, mp_pose, mmpose_model, openpose_model, yolo_model,
                  save_dir="results_skeletons"):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W,H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(save_dir, os.path.basename(video_path).replace(".mp4","_skeleton.mp4"))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))

    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
                  desc=f"Processing {os.path.basename(video_path)}"):
        ok, frame = cap.read()
        if not ok: break

        # 1️⃣ YOLO 检测所有人
        person_boxes = detect_person_yolo(yolo_model, frame)

        # 2️⃣ Mediapipe → 每人右腕（绿色）
        wrists_mp = get_wrists_mediapipe(frame, mp_pose, person_boxes)
        for w in wrists_mp:
            draw_point(frame, w, (0,255,0))

        # 3️⃣ MMPose → 每人右腕（蓝色）
        #wrists_mm = get_wrists_mmpose(frame, mmpose_model, person_boxes)
        #for w in wrists_mm:
        #    draw_point(frame, np.append(w[:2],1.0), (255,0,0))

        # 4️⃣ OpenPose → 多人右腕（白色）
        #wrists_op = get_wrist_openpose(frame, openpose_model)
        #for w in wrists_op:
        #    draw_point(frame, w, (255,255,255))

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ saved: {out_path}")

##########################
# 主程序
##########################

def main():
    # ========= 1️⃣ 读取 config.yaml =========
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    # ========= 2️⃣ 定位 annotation.json（在 annotation 目录中） =========
    ann_path = os.path.join(paths["report_root"], "annotation.json")
    data_root = paths["data_root"]
    save_dir = os.path.join(paths["report_root"], "results_skeletons")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"❌ 找不到 annotation.json: {ann_path}")

    # ========= 3️⃣ 加载 annotation.json 内容 =========
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # ========= 4️⃣ 初始化模型 =========
    print("🧩 Loading YOLO / Mediapipe / MMPose / OpenPose ...")
    yolo_model = init_yolo()
    mp_pose = init_mediapipe()
    mmpose_model = None #init_mmpose()
    openpose_model = None #init_openpose()

    # ========= 5️⃣ 根据 annotation.json 遍历 subject 和视频 =========
    for subj in annotations:
        subject_id = subj["subject"]
        print(f"\n=== 🎥 Subject: {subject_id} ===")

        for idx, vid_path in enumerate(subj["video_vector"]):
            # 替换 ${paths.data_root} -> 实际数据根目录
            video_path = vid_path.replace("${paths.data_root}", data_root)

            if not os.path.exists(video_path):
                print(f"⚠️ 找不到视频: {video_path}")
                continue

            print(f"▶️  [{idx+1}/{len(subj['video_vector'])}] {os.path.basename(video_path)}")
            process_video(
                video_path,
                mp_pose,
                mmpose_model,
                openpose_model,
                yolo_model,
                save_dir=save_dir
            )

    print(f"\n✅ 全部视频处理完成！结果已保存到: {save_dir}")
if __name__ == "__main__":
    main()