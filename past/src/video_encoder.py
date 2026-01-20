import torch
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from decord import VideoReader, cpu
import numpy as np
import cv2
import torch.nn.functional as F

class VideoActionEncoder:
    def __init__(self, model_name="MCG-NJU/videomae-base-finetuned-kinetics", device="cuda"):
        self.device = device
        self.extractor = VideoMAEFeatureExtractor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name).to(device).eval()

    def _sample_video(self, video_path, num_frames=16):
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        frames = np.stack([vr[i].asnumpy() for i in idxs])
        frames = torch.tensor(frames).permute(0, 3, 1, 2)  # [T,C,H,W]
        return list(frames)

    def encode_video(self, video_path):
        frames = self._sample_video(video_path)
        inputs = self.extractor(frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        logits = torch.softmax(out.logits, dim=-1)
        emb = F.normalize(logits, dim=-1)
        return emb
    def encode_person(self, video_path, bbox, num_frames=16):
        x1, y1, x2, y2 = map(int, bbox)
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        frames = []
        for i in idxs:
            img = vr[i].asnumpy()
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224, 224))
            frames.append(crop)
        frames = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)  # [T,C,H,W]
        inputs = self.extractor(list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        logits = torch.softmax(out.logits, dim=-1)
        emb = F.normalize(logits, dim=-1)
        return emb
    def predict_action(self, video_path, box=None):
        from transformers import VideoMAEImageProcessor
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        idxs = np.linspace(0, total - 1, 16).astype(int)
        frames = []
        for i in idxs:
            img = vr[i].asnumpy()
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                img = img[y1:y2, x1:x2]
            img = cv2.resize(img, (224, 224))
            frames.append(img)

        frames = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)
        inputs = self.extractor(list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        probs = torch.softmax(out.logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()
        return pred_id, probs.max().item()