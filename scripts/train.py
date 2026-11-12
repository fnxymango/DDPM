"""Stage 5/6: train DDPM (Algorithm 1) and sample (Algorithm 2)."""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torchvision.utils import save_image

from ddpm.data import make_dataset
from ddpm.diffusion import Diffusion
from ddpm.unet import UNet

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=["mnist", "pneumonia"])
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--ema", type=float, default=0.9995)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sample_every", type=int, default=5000)
p.add_argument("--out", required=True)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
json.dump(vars(args), open(f"{args.out}/config.json", "w"), indent=2)
torch.manual_seed(args.seed)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ds = make_dataset(args.dataset, "train")
X = ds.tensors[0].to(dev)  # whole set on GPU (CPU on this server is heavily shared)
model = UNet().to(dev)
ema = copy.deepcopy(model).eval().requires_grad_(False)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, fused=True)
model_c = torch.compile(model)
diff = Diffusion(device=dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"train images={len(ds)} params={n_params / 1e6:.2f}M", flush=True)


def sample_grid(tag):
    torch.manual_seed(123)
    with torch.autocast("cuda", torch.bfloat16):
        x, _ = diff.sample(ema, (64, 1, 28, 28), dev)
    x = x.float()
    save_image((x + 1) / 2, f"{args.out}/samples_{tag}.png", nrow=8)


log = open(f"{args.out}/loss.csv", "w")
log.write("step,loss,elapsed_s\n")
step, t0, run = 0, time.time(), torch.zeros((), device=dev)
while step < args.steps:
    for idx in torch.randperm(len(X), device=dev).split(args.bs):
        if len(idx) < args.bs:
            continue
        x0 = X[idx]
        with torch.autocast("cuda", torch.bfloat16):
            loss = diff.loss(model_c, x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - args.ema)
        step += 1
        run += loss.detach().float()
        if step % 100 == 0:
            r = run.item() / 100
            log.write(f"{step},{r:.5f},{time.time() - t0:.1f}\n")
            log.flush()
            if step % 1000 == 0:
                print(f"step {step} loss {r:.4f} {time.time() - t0:.0f}s", flush=True)
            run.zero_()
        if step % args.sample_every == 0 or step == args.steps:
            sample_grid(f"step{step:06d}")
            ck = {"model": model.state_dict(), "ema": ema.state_dict(), "step": step}
            torch.save(ck, f"{args.out}/ckpt.pt")
            torch.save({"ema": ck["ema"], "step": step}, f"{args.out}/ema_step{step:06d}.pt")
        if step >= args.steps:
            break
print(f"done {time.time() - t0:.0f}s", flush=True)
