"""Real (test) vs generated, 5 each, shown individually and enlarged. No cherry-picking: first 5 of each."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ddpm.data import make_dataset

run = sys.argv[1] if len(sys.argv) > 1 else "runs/pneumonia/eval_ema_step020000"
n = 5
gen = torch.load(f"{run}/generated_1000.pt")[:n]
torch.manual_seed(0)
test = make_dataset("pneumonia", "test").tensors
idx = torch.randperm(len(test[0]))[:n]
real, lab = test[0][idx], test[1][idx]
fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 7))
for j in range(n):
    ax[0, j].imshow((real[j, 0] + 1) / 2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[0, j].set_title(f"real #{j+1} ({'pneumonia' if lab[j] else 'normal'})", fontsize=11)
    ax[1, j].imshow((gen[j, 0] + 1) / 2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[1, j].set_title(f"generated #{j+1}", fontsize=11)
for a in ax.flat:
    a.axis("off")
fig.text(0.01, 0.73, "real (test)", rotation=90, fontsize=13, va="center")
fig.text(0.01, 0.28, "generated", rotation=90, fontsize=13, va="center")
plt.tight_layout(rect=(0.03, 0, 1, 1))
plt.savefig(f"{run}/compare_real_vs_gen_5.png", dpi=110)
print("saved")
