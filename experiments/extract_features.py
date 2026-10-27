"""Extract frozen ResNet-18 features once per dataset for the Table 12 rerun.

Protocol mirrors the original deep-baseline script (GGPDatast/fewshot_baselines.py):
torchvision ResNet-18 ImageNet1K_V1, fc removed (global average pooling, 512-d),
grayscale -> /255 -> bilinear 224x224 -> 3-channel -> ImageNet normalization.
Features are cached per image path so multi-split evaluation is cheap.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

import adle_core as ac

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "features")
os.makedirs(OUT, exist_ok=True)

DATA_ROOT = os.path.abspath(ac._REPO) + os.sep + "data"
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m = torch.nn.Sequential(*list(m.children())[:-1])
    m.eval()
    return m


@torch.no_grad()
def embed_batch(model, imgs):
    # imgs: list of 2D float arrays in [0,1]
    t = torch.from_numpy(np.stack(imgs)).unsqueeze(1)          # B,1,H,W
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)
    t = torch.stack([NORM(c) for c in t])
    f = model(t).squeeze(-1).squeeze(-1)
    return f.numpy()


def dataset_files(name):
    """Yield (key, image) for every candidate image of a dataset."""
    import cv2
    base = os.path.join(DATA_ROOT, {"ORL": "ORL", "AR": "AR", "Yale": "Yale",
                                    "YaleB": "Yale B", "UMIST": "UMIST", "PIE": "PIE",
                                    "COIL20": "coil-20-proc", "COIL100": "coil-100"}[name])
    if name == "ORL":
        data = np.load(os.path.join(base, "olivetti.npz"))
        for i, img in enumerate(data["X"]):
            yield f"olivetti#{i}", np.asarray(img).reshape(64, 64)
        return
    if name == "AR":
        for sd in sorted(os.listdir(base)):
            for f in sorted(os.listdir(os.path.join(base, sd))):
                if f.endswith(".jpg"):
                    p = os.path.join(base, sd, f)
                    yield p, cv2.resize(cv2.imread(p, 0), (32, 32)) / 255.0
        return
    if name == "Yale":
        for sd in sorted(os.listdir(base), key=int):
            for f in sorted(os.listdir(os.path.join(base, sd))):
                if f.endswith(".bmp"):
                    p = os.path.join(base, sd, f)
                    yield p, cv2.resize(cv2.imread(p, 0), (64, 64)) / 255.0
        return
    if name == "YaleB":
        for sd in sorted(d for d in os.listdir(base) if d.startswith("yaleB")):
            for f in sorted(x for x in os.listdir(os.path.join(base, sd)) if x.endswith(".pgm")):
                p = os.path.join(base, sd, f)
                img = cv2.imread(p, 0)
                if img is not None:
                    yield p, cv2.resize(img, (64, 64)) / 255.0
        return
    if name == "UMIST":
        for sd in sorted(d for d in os.listdir(base) if d.startswith("1")):
            for f in sorted(x for x in os.listdir(os.path.join(base, sd)) if x.endswith(".pgm")):
                p = os.path.join(base, sd, f)
                yield p, cv2.resize(cv2.imread(p, 0), (64, 64)) / 255.0
        return
    if name == "PIE":
        sub = os.path.join(base, "Pose05_64x64_files")
        for i in range(1, 68 * 49 + 1):
            p = os.path.join(sub, f"{i}.jpg")
            if os.path.exists(p):
                yield p, cv2.imread(p, 0) / 255.0
        return
    if name in ("COIL20", "COIL100"):
        sub = base
        for f in sorted(os.listdir(sub)):
            if f.endswith(".png"):
                p = os.path.join(sub, f)
                yield p, cv2.resize(cv2.imread(p, 0), (64, 64)) / 255.0
        return
    raise ValueError(name)


def main():
    names = sys.argv[1:] or ["ORL", "Yale", "UMIST", "AR", "YaleB", "COIL20", "COIL100", "PIE"]
    model = build_model()
    for name in names:
        out = os.path.join(OUT, f"{name}.npz")
        if os.path.exists(out):
            print(f"{name}: cached, skip", flush=True)
            continue
        keys, feats = [], []
        batch = []
        for key, img in dataset_files(name):
            keys.append(key)
            batch.append(np.asarray(img, dtype=np.float32))
            if len(batch) == 128:
                feats.append(embed_batch(model, batch))
                batch = []
                print(f"{name}: {len(keys)} done", flush=True)
        if batch:
            feats.append(embed_batch(model, batch))
        F = np.vstack(feats)
        np.savez_compressed(out, keys=np.array(keys), features=F)
        print(f"{name}: saved {F.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
