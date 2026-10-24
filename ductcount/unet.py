"""U-Net duct-centre detector trained on Ambre's point marks.

Target = Gaussian heatmap at each duct centre (labels completed across overlapping fields),
prediction = local maxima of the heatmap. One peak per duct, so touching ducts that Ambre
marked once are learned as one."""
import argparse, os, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .config import path
from .register import load_small, fov_mask, SCALE

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SIGMA = 6        # px at SCALE
CROP = 384
MODELS = path("models")


def heatmap(shape, pts, sigma=SIGMA):
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    hm = np.zeros(shape, np.float32)
    for x, y in pts:
        hm = np.maximum(hm, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)))
    return hm.astype(np.float32)


# Reinhard colour normalisation: per-image LAB mean/std (inside the field of view) mapped to the
# dataset average, to remove staining / exposure differences between mice and sessions.
LAB_TARGET = np.array([158.6, 130.44, 130.23]), np.array([18.58, 5.48, 4.83])


def normalize(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = fov_mask(im, 31) > 0
    mu, sd = lab[m].mean(0), lab[m].std(0) + 1e-3
    lab = (lab - mu) / sd * LAB_TARGET[1] + LAB_TARGET[0]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out[~m] = 0
    return out


def to_tensor(im):
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32))


def color_jitter(im):
    """Stain variation: random per-channel gain/offset in OD space, plus brightness."""
    od = -np.log((im.astype(np.float32) + 1) / 256)
    od = od * np.random.uniform(0.7, 1.3, 3) + np.random.uniform(-0.08, 0.08, 3)
    out = 256 * np.exp(-od) - 1
    out *= np.random.uniform(0.85, 1.15)
    return np.clip(out, 0, 255).astype(np.uint8)


class Data(torch.utils.data.Dataset):
    """items: (path, all duct points) or (path, all duct points, atypical points, classified).
    With atypical points the target has a second channel (atypical ducts only), supervised only on
    fields whose ducts were classified (classified=True); elsewhere that channel is ignored."""
    def __init__(self, items, n=256, norm=False):
        self.items = []
        for it in items:
            im = load_small(it[0])
            im = normalize(im) if norm else im
            hm = np.stack([heatmap(im.shape[:2], np.asarray(q).reshape(-1, 2) * SCALE) for q in it[1:3]], -1)
            mask = np.array([1.0] + ([float(it[3])] if len(it) > 2 else []), np.float32)
            self.items.append((im, hm, mask))
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        im, hm, mask = random.choice(self.items)
        H, W = hm.shape[:2]
        y, x = random.randint(0, H - CROP), random.randint(0, W - CROP)
        im, hm = im[y:y + CROP, x:x + CROP], hm[y:y + CROP, x:x + CROP]
        k = random.randint(0, 3)
        im, hm = np.rot90(im, k), np.rot90(hm, k)
        if random.random() < 0.5:
            im, hm = im[:, ::-1], hm[:, ::-1]
        im = color_jitter(np.ascontiguousarray(im))
        return to_tensor(im), torch.from_numpy(np.ascontiguousarray(hm).transpose(2, 0, 1).copy()), torch.from_numpy(mask)


def focal(logits, hm, mask=None, a=2, b=4):
    """CenterNet focal loss on Gaussian targets. Each channel is normalised by its own number of
    positives; mask (B x C) switches channels off for samples where they are not annotated."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = hm.ge(0.99).float()
    l = -((1 - p) ** a) * torch.log(p) * pos - ((1 - hm) ** b) * (p ** a) * torch.log(1 - p) * (1 - pos)
    if mask is not None:
        l, pos = l * mask[:, :, None, None], pos * mask[:, :, None, None]
    return sum(l[:, c].sum() / pos[:, c].sum().clamp(min=1) for c in range(l.shape[1]))


def make_model(weights="imagenet", classes=1):
    """channel 0 = duct centres; optional channel 1 = atypical duct centres"""
    m = smp.Unet("resnet34", encoder_weights=weights, in_channels=3, classes=classes)
    torch.nn.init.constant_(m.segmentation_head[0].bias, -4.6)  # start at p~0.01 (CenterNet init)
    return m


def train(items, epochs=40, out=None, seed=0, log=print, norm=False, init=None, lr=1e-3):
    """init: checkpoint to fine-tune from (its output head is reused when the number of classes matches)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = make_model(classes=2 if len(items[0]) > 2 else 1)
    if init:
        ck = torch.load(init, map_location="cpu")
        sd = ck["state_dict"] if "state_dict" in ck else ck
        own = model.state_dict()
        model.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
        if own["segmentation_head.0.weight"].shape != sd["segmentation_head.0.weight"].shape:
            with torch.no_grad():  # new atypical channel: start from the duct channel
                model.segmentation_head[0].weight[:1] = sd["segmentation_head.0.weight"]
                model.segmentation_head[0].bias[:1] = sd["segmentation_head.0.bias"]
    model = model.to(DEV)
    dl = torch.utils.data.DataLoader(Data(items, norm=norm), batch_size=8, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl))
    for ep in range(epochs):
        model.train(); tot = 0
        for x, y, mask in dl:
            x, y, mask = x.to(DEV), y.to(DEV), mask.to(DEV)
            loss = focal(model(x), y, mask)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"  epoch {ep + 1}/{epochs} loss {tot / len(dl):.3f}")
    model.norm = norm
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), norm=norm), out)
    return model


def load(path):
    ck = torch.load(path, map_location="cpu")
    sd, norm = (ck["state_dict"], ck["norm"]) if "state_dict" in ck else (ck, False)  # old checkpoints: raw weights
    m = make_model(None, classes=sd["segmentation_head.0.weight"].shape[0])
    m.load_state_dict(sd)
    m.norm = norm
    return m.to(DEV).eval()


def predict_heatmap(model, path):
    return predict_maps(model, path)[0]


@torch.no_grad()
def predict_maps(model, path):
    """C x H x W probability maps at SCALE (channel 0 = ducts, 1 = atypical if the model has it)."""
    im = load_small(path)
    H, W = im.shape[:2]
    x = to_tensor(normalize(im) if getattr(model, "norm", False) else im)[None]
    ph, pw = (-H) % 32, (-W) % 32
    x = F.pad(x, (0, pw, 0, ph), mode="reflect").to(DEV)
    # 8-fold test-time augmentation (rotations + flip)
    acc = 0
    for k in range(4):
        for flip in (False, True):
            xi = torch.rot90(x, k, (2, 3))
            xi = torch.flip(xi, (3,)) if flip else xi
            y = torch.sigmoid(model(xi))
            y = torch.flip(y, (3,)) if flip else y
            acc = acc + torch.rot90(y, -k, (2, 3))
    hm = (acc / 8)[0, :, :H, :W].cpu().numpy()
    hm[:, fov_mask(im, 31) == 0] = 0
    return hm.astype(np.float32)


def peaks(hm, thr=0.3, min_dist=8):
    """Local maxima above thr, returned as full-res (x, y)."""
    mx = cv2.dilate(hm, np.ones((2 * min_dist + 1,) * 2, np.uint8))
    ys, xs = np.where((hm == mx) & (hm > thr))
    return np.stack([xs, ys], 1).astype(float) / SCALE
