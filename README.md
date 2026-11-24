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

To evaluate a saved checkpoint without retraining:

```bash
python act.py --eval-ckpt act.pt              # add --ensemble-m 0.01 to change the ensembling weight
```

## Loss curves

![loss curves](assets/loss_curves.png)

Eval metric: L1 on held-out episodes (every 7th episode) with z = 0, as at test time.

## Temporal ensembling

At inference the policy is queried at every timestep instead of once every k = 20 steps. Each query predicts
actions for t..t+k-1, so the action for timestep t has been predicted by up to k queries. `TemporalEnsembler`
averages these predictions with weights w_i = exp(-m·i), where i = 0 is the oldest prediction (m = 0.01, as in the
ACT paper). This gives smoother actions than executing each chunk open loop.

`rollout_eval` replays the held-out episodes step by step and compares three ways to turn chunks into actions:

| mode | queries | action executed |
| --- | --- | --- |
| `chunked` | every k steps | the whole chunk, in order |
| `first` | every step | the first action of the chunk |
| `ensemble` | every step | temporal ensemble of all predictions for this step |

It reports L1 against the recorded actions and jitter (mean |a_t − a_{t−1}|), both in normalized units. It runs
after training and with `--eval-ckpt`. Observations come from the recording (open loop), so it measures how
accurate and smooth the actions are, not whether the task succeeds.

## Rollout

_Coming soon._
