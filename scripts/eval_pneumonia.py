"""Stage 6: check generated PneumoniaMNIST samples.

1) real vs generated grid
2) memorization: nearest-train-neighbor L2 of generated vs. of real test images
3) intensity statistics (per-image mean / std) generated vs real test
4) class mix: small CNN trained on real train (normal/pneumonia) applied to generated
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from ddpm.data import make_dataset
from ddpm.diffusion import Diffusion
from ddpm.unet import UNet

run = sys.argv[1] if len(sys.argv) > 1 else "runs/pneumonia"
ckpt = sys.argv[2] if len(sys.argv) > 2 else f"{run}/ckpt.pt"
tag = os.path.splitext(os.path.basename(ckpt))[0]
N = 1000
out = f"{run}/eval_{tag}"
os.makedirs(out, exist_ok=True)
dev = "cuda"
torch.manual_seed(0)

train = make_dataset("pneumonia", "train").tensors
test = make_dataset("pneumonia", "test").tensors
val = make_dataset("pneumonia", "val").tensors

# --- generate
model = UNet().to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu")["ema"])
model.eval()
diff = Diffusion(device=dev)
gen = []
for _ in range(N // 250):
    with torch.autocast("cuda", torch.bfloat16):
        x, _ = diff.sample(model, (250, 1, 28, 28), dev)
    gen.append(x.float().cpu())
gen = torch.cat(gen)
torch.save(gen, f"{out}/generated_{N}.pt")

# --- 1) grids
real = test[0][torch.randperm(len(test[0]))[:64]]
save_image((real + 1) / 2, f"{out}/grid_real_test.png", nrow=8)
save_image((gen[:64] + 1) / 2, f"{out}/grid_generated.png", nrow=8)


# --- 2) nearest neighbor to train
def nn_dist(q, ref):
    q, ref = q.flatten(1).to(dev), ref.flatten(1).to(dev)
    d = torch.cdist(q, ref)
    v, i = d.min(1)
    return v.cpu() / q.shape[1] ** 0.5, i.cpu()  # RMS per pixel


tr = train[0]
d_gen, i_gen = nn_dist(gen, tr)
d_test, _ = nn_dist(test[0], tr)
order = d_gen.argsort()[:8]
pairs = torch.cat([gen[order], tr[i_gen[order]]])
save_image((pairs + 1) / 2, f"{out}/nn_closest8_gen_top_train_bottom.png", nrow=8)

# --- 3) intensity stats
def stats(x):
    x = (x + 1) / 2
    return x.mean((1, 2, 3)).numpy(), x.std((1, 2, 3)).numpy()


gm, gs = stats(gen)
tm, ts = stats(test[0])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].hist(tm, 40, alpha=.6, density=True, label="real test"); ax[0].hist(gm, 40, alpha=.6, density=True, label="generated")
ax[0].set_title("per-image mean intensity"); ax[0].legend()
ax[1].hist(ts, 40, alpha=.6, density=True); ax[1].hist(gs, 40, alpha=.6, density=True); ax[1].set_title("per-image std")
ax[2].hist(d_test.numpy(), 40, alpha=.6, density=True, label="real test -> train")
ax[2].hist(d_gen.numpy(), 40, alpha=.6, density=True, label="generated -> train")
ax[2].set_title("nearest-train RMS distance"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{out}/eval_hists.png", dpi=120)


# --- 4) class-mix via small classifier
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 49, 2))

    def forward(self, x):
        return self.f(x)


clf = CNN().to(dev)
opt = torch.optim.Adam(clf.parameters(), 1e-3)
X, Y = train[0].to(dev), train[1].to(dev)
w = torch.tensor([1 / (Y == 0).float().mean(), 1 / (Y == 1).float().mean()], device=dev)
best, best_state = 0, None
for ep in range(20):
    clf.train()
    for idx in torch.randperm(len(X), device=dev).split(128):
        loss = F.cross_entropy(clf(X[idx]), Y[idx], weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        pv = clf(val[0].to(dev)).argmax(1).cpu()
    bal = 0.5 * sum((pv[val[1] == c] == c).float().mean().item() for c in (0, 1))
    if bal > best:
        best, best_state = bal, {k: v.clone() for k, v in clf.state_dict().items()}
clf.load_state_dict(best_state); clf.eval()
with torch.no_grad():
    pt = clf(test[0].to(dev)).argmax(1).cpu()
    pg = clf(gen.to(dev)).argmax(1).cpu()
    conf_g = F.softmax(clf(gen.to(dev)), 1).max(1).values.cpu()
    conf_t = F.softmax(clf(test[0].to(dev)), 1).max(1).values.cpu()
test_bal = 0.5 * sum((pt[test[1] == c] == c).float().mean().item() for c in (0, 1))

res = {
    "n_generated": N,
    "train_pneumonia_frac": round(train[1].float().mean().item(), 3),
    "test_pneumonia_frac_true": round(test[1].float().mean().item(), 3),
    "test_pneumonia_frac_pred": round(pt.float().mean().item(), 3),
    "generated_pneumonia_frac_pred": round(pg.float().mean().item(), 3),
    "clf_val_balanced_acc": round(best, 3),
    "clf_test_balanced_acc": round(test_bal, 3),
    "clf_mean_confidence_test": round(conf_t.mean().item(), 3),
    "clf_mean_confidence_generated": round(conf_g.mean().item(), 3),
    "nn_rms_test_to_train_median": round(d_test.median().item(), 4),
    "nn_rms_gen_to_train_median": round(d_gen.median().item(), 4),
    "nn_rms_gen_to_train_min": round(d_gen.min().item(), 4),
    "nn_rms_test_to_train_min": round(d_test.min().item(), 4),
    "gen_below_test_p1": round((d_gen < d_test.quantile(0.01)).float().mean().item(), 3),
    "mean_intensity_test": round(float(tm.mean()), 4),
    "mean_intensity_gen": round(float(gm.mean()), 4),
    "std_intensity_test": round(float(ts.mean()), 4),
    "std_intensity_gen": round(float(gs.mean()), 4),
}
json.dump(res, open(f"{out}/eval.json", "w"), indent=2)
print(json.dumps(res, indent=2))
