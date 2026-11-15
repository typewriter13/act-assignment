# ACT from scratch

ACT ([Action Chunking with Transformers](https://arxiv.org/abs/2304.13705)) implemented from scratch in PyTorch,
trained on the ABC-130k sim task **put_the_plastic_bottles_in_the_bin** (bimanual, 3 cameras, 14-D state/action).

Everything (dataloader, model, training loop, eval) is in [`act.py`](act.py).

## Run

```bash
pip install -r requirements.txt
python act.py --steps 2000      # downloads 35 episodes, caches them, trains, evaluates
```

Outputs `act.pt` and `assets/loss_curves.png`. Add `--no-wandb` to skip Weights & Biases logging.

## Loss curves

![loss curves](assets/loss_curves.png)

Eval metric: L1 on held-out episodes (every 7th episode) with z = 0, as at test time.

## Rollout

_Coming soon._
