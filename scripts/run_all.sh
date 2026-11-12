#!/bin/bash
# Full pipeline: forward vis -> MNIST -> PneumoniaMNIST -> eval
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs/logs
$PY scripts/forward_vis.py > runs/logs/forward.log 2>&1
$PY scripts/train.py --dataset mnist --steps 20000 --out runs/mnist > runs/logs/mnist.log 2>&1
$PY scripts/train.py --dataset pneumonia --steps 20000 --out runs/pneumonia > runs/logs/pneumonia.log 2>&1
for s in 005000 020000; do
  $PY scripts/eval_pneumonia.py runs/pneumonia runs/pneumonia/ema_step$s.pt > runs/logs/eval_$s.log 2>&1
done
echo ALL_DONE >> runs/logs/pneumonia.log
