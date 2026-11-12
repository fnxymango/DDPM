"""Stage 4: noise schedule + q_sample visualization."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ddpm.data import load_images, to_tensor
from ddpm.diffusion import Diffusion

out = "runs/forward"
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
d = Diffusion()
ts = [0, 50, 100, 200, 400, 600, 800, 999]

fig, axes = plt.subplots(2, len(ts), figsize=(2 * len(ts), 4.4))
for row, name in enumerate(["mnist", "pneumonia"]):
    imgs, _ = load_images(name, "train")
    x0 = to_tensor(imgs[:1])
    noise = torch.randn_like(x0)
    for j, t in enumerate(ts):
        xt = d.q_sample(x0, torch.tensor([t]), noise)
        ax = axes[row, j]
        ax.imshow(xt[0, 0].clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"t={t + 1}", fontsize=9)
        ax.axis("off")
plt.tight_layout()
plt.savefig(f"{out}/q_sample_grid.png", dpi=120)

ab = d.alpha_bars
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(ab.sqrt(), label="sqrt(alpha_bar) (signal)")
ax.plot((1 - ab).sqrt(), label="sqrt(1 - alpha_bar) (noise)")
ax.set_xlabel("t")
ax.legend()
plt.tight_layout()
plt.savefig(f"{out}/schedule.png", dpi=120)

with open(f"{out}/schedule.txt", "w") as f:
    for t in ts:
        f.write(f"t={t + 1}\tabar={ab[t]:.5f}\tsignal={ab[t].sqrt():.4f}\tnoise={(1 - ab[t]).sqrt():.4f}\n")
print(open(f"{out}/schedule.txt").read())
