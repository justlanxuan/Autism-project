import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils import IMUPairsDataset
from src.imu_encoder import IMUEncoder

import torch, gc
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
gc.collect()

batch_size = 4
hidden_dim = 64
embed_dim = 128


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def train_imu_action(
    json_path,
    device="cuda",
    batch_size=8,
    lr=1e-4,
    epochs=30,
    embed_dim=256,
    save_path="./checkpoints/imu_action.pth"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = IMUPairsDataset(json_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=False, persistent_workers=False)
    n_action = len(dataset.actions)
    print(f"[INFO] Loaded {len(dataset)} samples, {n_action} actions.")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Using device: {device} ({device_name})")

    model = IMUEncoder(input_dim=9, hidden_dim=128, embed_dim=embed_dim).to(device)
    classifier = nn.Linear(embed_dim, n_action).to(device)
    resume_path = "./checkpoints/imu_action.pth"
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["encoder"])
        classifier.load_state_dict(ckpt["head"])
        print(f"[INFO] Resumed training from {resume_path}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train(); classifier.train()
        tot_loss, tot_acc = 0, 0
        for imu, lbl in tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]"):
            imu, lbl = imu.to(device), lbl.to(device)
            emb = model(imu)
            pred = classifier(emb)
            loss = criterion(pred, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (pred.argmax(1) == lbl).float().mean().item()
            tot_loss += loss.item(); tot_acc += acc

        print(f"Epoch{epoch:02d}: loss={tot_loss/len(loader):.4f}, acc={tot_acc/len(loader)*100:.2f}%")

    torch.save({"encoder": model.state_dict(), "head": classifier.state_dict(),
                "act2id": dataset.act2id},
               save_path)
    print(f"Saved at {save_path}")


if __name__ == "__main__":
    cfg_path = "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    json_path = os.path.join(cfg["report_root"], "annotation_pairs.json")
    train_imu_action(
        json_path=json_path,
        device=cfg["device"],
        batch_size=cfg["batch_size"],
        lr = float(cfg.get("lr", 1e-4)),
        epochs=cfg["num_epochs"],
        embed_dim=cfg["embedding_dim"],
        save_path=os.path.join(cfg["save_dir"], "imu_action.pth"),
    )