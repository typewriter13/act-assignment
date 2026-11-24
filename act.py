"""ACT (Action Chunking with Transformers) from scratch on ABC-130k sim data.

Single file: data loading, model, training loop and eval.
    python act.py                      # download 35 episodes, cache, train 2000 steps, eval
    python act.py --steps 2000 --no-wandb
"""
import argparse
import glob
import io
import math
import os
import pickle
import time
from collections import deque

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import wandb
from huggingface_hub import HfApi, RepoFolder, hf_hub_download
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.ops.misc import FrozenBatchNorm2d

HF_REPO = "XDOF/ABC-130k"
TASK = "data/sim/put_the_plastic_bottles_in_the_bin"
STATE_TOPICS = ["/left-arm-state", "/left-ee-state", "/right-arm-state", "/right-ee-state"]
ACTION_TOPICS = ["/left-arm-action", "/left-ee-action", "/right-arm-action", "/right-ee-action"]
CAMERAS = ["/top-camera", "/left-wrist-camera", "/right-wrist-camera"]
CHUNK = 20


# ----------------------------------------------------------------------------- data

def download(n, out_dir):
    api = HfApi()
    folders = sorted(e.path for e in api.list_repo_tree(HF_REPO, path_in_repo=TASK, repo_type="dataset")
                     if isinstance(e, RepoFolder))
    for i, folder in enumerate(folders[:n]):
        hf_hub_download(HF_REPO, f"{folder}/episode.mcap", repo_type="dataset", local_dir=out_dir)
        print(f"downloaded [{i + 1}/{n}] {folder}")


def get_time(proto):
    return proto.timestamp.seconds + proto.timestamp.nanos * 1e-9


def load_episode(path):
    """joint states, actions (T, 14) and their timestamps from one .mcap"""
    data, times = {}, []
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, proto in reader.iter_decoded_messages():
            if channel.topic in STATE_TOPICS or channel.topic in ACTION_TOPICS:
                data.setdefault(channel.topic, []).append(list(proto.position))
            if channel.topic == "/left-arm-state":
                times.append(get_time(proto))  # one time per timestep
    states = np.concatenate([np.array(data[t]) for t in STATE_TOPICS], axis=1)
    actions = np.concatenate([np.array(data[t]) for t in ACTION_TOPICS], axis=1)
    return states, actions, np.array(times)


def load_camera(path, topic):
    """decode the H.264 stream of one camera -> frames (N, H, W, 3), timestamps (N,)"""
    codec = av.CodecContext.create("h264", "r")
    frames, times = [], []

    def decode(packets):
        for packet in packets:
            for frame in codec.decode(packet):
                frames.append(frame.to_ndarray(format="rgb24"))

    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, proto in reader.iter_decoded_messages(topics=[topic]):
            times.append(get_time(proto))
            decode(codec.parse(proto.data))
    decode(codec.parse(None))  # flush leftovers
    decode([None])
    assert len(frames) == len(times), "frame count != message count"
    return np.stack(frames), np.array(times)


def build_cache(mcap_path, out_dir):
    """Decode one episode once; save states, actions and JPEG frames aligned to each state."""
    if os.path.exists(os.path.join(out_dir, "frames.pkl")):
        return
    os.makedirs(out_dir, exist_ok=True)
    states, actions, state_t = load_episode(mcap_path)
    frames = [[] for _ in range(len(states))]  # frames[t] = [top, left wrist, right wrist] JPEG bytes
    for cam in CAMERAS:
        imgs, cam_t = load_camera(mcap_path, cam)
        nearest = np.abs(cam_t[None, :] - state_t[:, None]).argmin(axis=1)
        for t, j in enumerate(nearest):
            buf = io.BytesIO()
            Image.fromarray(imgs[j]).save(buf, format="JPEG", quality=95)
            frames[t].append(buf.getvalue())
    np.save(os.path.join(out_dir, "states.npy"), states)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    tmp = os.path.join(out_dir, "frames_tmp.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(frames, f)
    os.replace(tmp, os.path.join(out_dir, "frames.pkl"))


class ACTDataset(Dataset):
    """sample = (images (3, 3, H, W), state (14,), next CHUNK actions (CHUNK, 14), is_pad (CHUNK,))"""

    def __init__(self, cache_dirs, chunk=CHUNK, stats=None):
        self.chunk = chunk
        self.states, self.actions, self.frames = [], [], []
        for d in cache_dirs:
            self.states.append(np.load(os.path.join(d, "states.npy")))
            self.actions.append(np.load(os.path.join(d, "actions.npy")))
            with open(os.path.join(d, "frames.pkl"), "rb") as f:
                self.frames.append(pickle.load(f))
        self.index = [(e, t) for e, s in enumerate(self.states) for t in range(len(s))]

        if stats is None:  # normalization stats from this data (val uses the train stats)
            all_s, all_a = np.concatenate(self.states), np.concatenate(self.actions)
            stats = {"s_mean": all_s.mean(0), "s_std": all_s.std(0) + 1e-6,
                     "a_mean": all_a.mean(0), "a_std": all_a.std(0) + 1e-6}
        self.stats = stats

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        chunk = self.actions[e][t : t + self.chunk]
        n = len(chunk)
        is_pad = np.zeros(self.chunk, dtype=bool)
        if n < self.chunk:  # pad past the end of the episode with the last action
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self.chunk - n, axis=0)])
            is_pad[n:] = True
        s = self.stats
        state = (self.states[e][t] - s["s_mean"]) / s["s_std"]
        chunk = (chunk - s["a_mean"]) / s["a_std"]
        imgs = np.stack([np.array(Image.open(io.BytesIO(b)).convert("RGB")) for b in self.frames[e][t]])
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255
        return imgs, torch.from_numpy(state).float(), torch.from_numpy(chunk).float(), torch.from_numpy(is_pad)


# ----------------------------------------------------------------------------- model

def sine_pos_1d(n, d):
    pos = torch.arange(n).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (n, d)


def sine_pos_2d(h, w, d):
    ey, ex = sine_pos_1d(h, d // 2), sine_pos_1d(w, d // 2)
    pe = torch.cat([ey[:, None, :].expand(h, w, -1), ex[None, :, :].expand(h, w, -1)], dim=-1)
    return pe.reshape(h * w, d)  # (h*w, d)


class ImageEncoder(nn.Module):
    """images (B, N_cam, 3, H, W) in [0,1]  ->  tokens (B, T, d), pos (1, T, d)"""

    def __init__(self, d=512):
        super().__init__()
        self.d = d
        r18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                                          norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(r18.children())[:-2])  # drop avgpool + fc
        self.proj = nn.Conv2d(512, d, kernel_size=1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        B, N = images.shape[:2]
        x = (images.flatten(0, 1) - self.mean) / self.std  # (B*N, 3, H, W)
        f = self.proj(self.backbone(x))  # (B*N, d, h, w)
        h, w = f.shape[-2:]
        # cameras side by side: (B, d, h, N*w)
        f = f.view(B, N, self.d, h, w).permute(0, 2, 3, 1, 4).reshape(B, self.d, h, N * w)
        tokens = f.flatten(2).transpose(1, 2)  # (B, h*N*w, d)
        pos = sine_pos_2d(h, N * w, self.d).to(f.device)[None]
        return tokens, pos


class CVAEEncoder(nn.Module):
    """state (B, S), actions (B, k, A), is_pad (B, k)  ->  z, mu, logvar  (each (B, latent))"""

    def __init__(self, state_dim, action_dim, k, d, heads, layers, ff, latent, dropout):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, d)
        self.action_proj = nn.Linear(action_dim, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.register_buffer("pos", sine_pos_1d(2 + k, d)[None])
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.to_latent = nn.Linear(d, 2 * latent)

    def forward(self, state, actions, is_pad):
        B = state.shape[0]
        tokens = torch.cat([self.cls.expand(B, -1, -1), self.state_proj(state)[:, None],
                            self.action_proj(actions)], dim=1) + self.pos  # (B, 2+k, d)
        mask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=state.device), is_pad], dim=1)
        summary = self.encoder(tokens, src_key_padding_mask=mask)[:, 0]  # CLS output
        mu, logvar = self.to_latent(summary).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar


class ACT(nn.Module):
    def __init__(self, state_dim=14, action_dim=14, k=CHUNK, d=512, heads=8,
                 enc_layers=4, dec_layers=7, ff=3200, latent=32, dropout=0.1):
        super().__init__()
        self.latent = latent
        self.image_encoder = ImageEncoder(d)
        self.cvae = CVAEEncoder(state_dim, action_dim, k, d, heads, 4, ff, latent, dropout)
        self.latent_proj = nn.Linear(latent, d)
        self.state_proj = nn.Linear(state_dim, d)
        self.aux_pos = nn.Parameter(torch.randn(1, 2, d) * 0.02)  # tags for [z, state] tokens
        enc_layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers, enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.queries = nn.Embedding(k, d)  # k learned queries -> k actions
        self.head = nn.Linear(d, action_dim)

    def forward(self, images, state, actions=None, is_pad=None):
        B = state.shape[0]
        if actions is not None:  # training: z from the real future actions
            z, mu, logvar = self.cvae(state, actions, is_pad)
        else:  # inference: z = 0 (mean of the prior)
            z = torch.zeros(B, self.latent, device=state.device)
            mu = logvar = None
        img_tokens, img_pos = self.image_encoder(images)
        src = torch.cat([self.latent_proj(z)[:, None], self.state_proj(state)[:, None], img_tokens], dim=1)
        pos = torch.cat([self.aux_pos, img_pos], dim=1)
        memory = self.encoder(src + pos)
        out = self.decoder(self.queries.weight[None].expand(B, -1, -1), memory)
        return self.head(out), mu, logvar  # (B, k, action_dim)


def act_loss(pred, actions, is_pad, mu, logvar, kl_weight):
    real = ~is_pad
    l1 = ((pred - actions).abs() * real.unsqueeze(-1)).sum() / (real.sum() * pred.shape[-1])
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, l1.detach(), kl.detach()


# ----------------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, loader, device):
    """L1 on held-out episodes with z = 0, i.e. how the policy acts at test time."""
    model.eval()
    total, count = 0.0, 0.0
    for imgs, state, chunk, is_pad in loader:
        imgs, state, chunk, is_pad = (x.to(device) for x in (imgs, state, chunk, is_pad))
        pred, _, _ = model(imgs, state)
        real = ~is_pad
        total += ((pred - chunk).abs() * real.unsqueeze(-1)).sum().item()
        count += real.sum().item() * pred.shape[-1]
    model.train()
    return total / count


class TemporalEnsembler:
    """Temporal ensembling from the ACT paper (Zhao et al. 2023, Sec. 3.3).

    The policy is queried at every timestep. Each query predicts actions for t..t+k-1, so the action
    for timestep t has been predicted by up to k queries. They are averaged with weights
    w_i = exp(-m * i), i = 0 being the oldest prediction; smaller m mixes in new observations faster.
    Call reset() at the start of every episode.
    """

    def __init__(self, k, m=0.01):
        self.k, self.m = k, m
        self.reset()

    def reset(self):
        self.chunks = deque(maxlen=self.k)  # the last k predicted chunks, oldest first

    def __call__(self, chunk):
        """chunk (k, A) predicted at the current timestep -> action (A,) to execute now"""
        self.chunks.append(chunk)
        n = len(self.chunks)
        # chunks[j] was predicted n-1-j steps ago, so its prediction for now sits at index n-1-j
        preds = torch.stack([c[n - 1 - j] for j, c in enumerate(self.chunks)])  # (n, A)
        w = torch.exp(-self.m * torch.arange(n, dtype=preds.dtype, device=preds.device))
        return (w[:, None] * preds).sum(0) / w.sum()


@torch.no_grad()
def rollout_eval(model, ds, device, m=0.01, batch=32):
    """Replay each held-out episode step by step and turn the predicted chunks into one action per step:
      chunked  - query every k steps and execute the whole chunk (no ensembling)
      first    - query every step and execute only the first action of the chunk
      ensemble - query every step and use temporal ensembling
    Reports L1 to the recorded actions and jitter (mean |a_t - a_{t-1}|), in normalized units.
    Observations come from the recording (open loop), so this measures accuracy and smoothness,
    not closed-loop task success.
    """
    model.eval()
    k = model.queries.num_embeddings
    ens = TemporalEnsembler(k, m)
    s = ds.stats
    out = {mode: [] for mode in ("chunked", "first", "ensemble")}
    gts = []
    for e in range(len(ds.states)):
        idx = [i for i, (ep, _) in enumerate(ds.index) if ep == e]
        # observations are recorded, so every chunk of the episode can be predicted in batches up front
        chunks = []
        for b in range(0, len(idx), batch):
            samples = [ds[i] for i in idx[b : b + batch]]
            imgs = torch.stack([x[0] for x in samples]).to(device)
            state = torch.stack([x[1] for x in samples]).to(device)
            chunks.append(model(imgs, state)[0].cpu())
        chunks = torch.cat(chunks)  # (T, k, A): chunks[t] predicts actions t..t+k-1
        T = len(chunks)
        out["chunked"].append(chunks[::k].flatten(0, 1)[:T])
        out["first"].append(chunks[:, 0])
        ens.reset()
        out["ensemble"].append(torch.stack([ens(chunks[t]) for t in range(T)]))
        gts.append(torch.from_numpy((ds.actions[e] - s["a_mean"]) / s["a_std"]).float())
    model.train()

    def jitter(seqs):
        return torch.cat([(x[1:] - x[:-1]).abs() for x in seqs]).mean().item()

    gt = torch.cat(gts)
    results = {f"{mode}_l1": (torch.cat(p) - gt).abs().mean().item() for mode, p in out.items()}
    results.update({f"{mode}_jitter": jitter(p) for mode, p in out.items()})
    results["recorded_jitter"] = jitter(gts)
    print(f"rollout on {len(gts)} held-out episodes (m={m}):")
    for mode in out:
        print(f"  {mode:9s} L1 {results[mode + '_l1']:.3f}  jitter {results[mode + '_jitter']:.4f}")
    print(f"  recorded  jitter {results['recorded_jitter']:.4f}")
    return results


# ----------------------------------------------------------------------------- train

def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # data: download -> cache -> train/val split by episode
    if not glob.glob(os.path.join(a.episodes_dir, "**/episode.mcap"), recursive=True):
        download(a.episodes, a.episodes_dir)
    for p in sorted(glob.glob(os.path.join(a.episodes_dir, "**/episode.mcap"), recursive=True)):
        build_cache(p, os.path.join(a.cache_dir, os.path.basename(os.path.dirname(p))))
    dirs = sorted(glob.glob(os.path.join(a.cache_dir, "*")))
    val_dirs = dirs[::7]  # every 7th episode held out
    train_dirs = [d for d in dirs if d not in val_dirs]
    train_ds = ACTDataset(train_dirs)
    val_ds = ACTDataset(val_dirs, stats=train_ds.stats)
    print(f"episodes: {len(train_dirs)} train, {len(val_dirs)} val | samples: {len(train_ds)} train, {len(val_ds)} val")
    train_loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True,
                              num_workers=a.num_workers, pin_memory=device == "cuda")
    val_loader = DataLoader(Subset(val_ds, range(0, len(val_ds), 20)), batch_size=a.batch,
                            num_workers=a.num_workers)

    model = ACT().to(device)
    if a.eval_ckpt:  # skip training: score a saved checkpoint with and without temporal ensembling
        ckpt = torch.load(a.eval_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        val_ds.stats = {key: ckpt[key] for key in ("s_mean", "s_std", "a_mean", "a_std")}
        rollout_eval(model, val_ds, device, a.ensemble_m)
        return

    wandb.init(project="act-from-scratch", config=vars(a), mode="disabled" if a.no_wandb else "online")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)

    hist = {"loss": [], "l1": [], "kl": []}
    val_hist = []
    step, t0 = 0, time.time()
    while step < a.steps:
        for imgs, state, chunk, is_pad in train_loader:
            imgs, state, chunk, is_pad = (x.to(device) for x in (imgs, state, chunk, is_pad))
            pred, mu, logvar = model(imgs, state, chunk, is_pad)
            loss, l1, kl = act_loss(pred, chunk, is_pad, mu, logvar, a.kl_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            log = {"loss": loss.item(), "l1": l1.item(), "kl": kl.item()}
            for k, v in log.items():
                hist[k].append(v)
            if step % a.val_every == 0 or step == a.steps:
                log["val_l1"] = evaluate(model, val_loader, device)
                val_hist.append((step, log["val_l1"]))
                print(f"  val L1 (z=0) at step {step}: {log['val_l1']:.3f}")
            wandb.log(log, step=step)
            if step % 10 == 0:
                print(f"step {step}/{a.steps}  loss {log['loss']:.3f}  l1 {log['l1']:.3f}  "
                      f"kl {log['kl']:.3f}  {time.time() - t0:.0f}s", flush=True)
            if step >= a.steps:
                break

    torch.save({"model": model.state_dict(), **train_ds.stats}, "act.pt")

    os.makedirs("assets", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ["loss", "l1", "kl"]):
        ax.plot(range(1, step + 1), hist[key], label="train")
        ax.set_title(key)
        ax.set_xlabel("update")
    axes[1].plot([s for s, _ in val_hist], [v for _, v in val_hist], "o-", label="val (z=0)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("assets/loss_curves.png", dpi=150)
    print("saved act.pt and assets/loss_curves.png")
    wandb.log({f"rollout/{key}": v for key, v in rollout_eval(model, val_ds, device, a.ensemble_m).items()})
    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--episodes", type=int, default=35, help="how many episodes to download")
    p.add_argument("--episodes-dir", default="episodes")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--ensemble-m", type=float, default=0.01, help="temporal ensembling weight decay m")
    p.add_argument("--eval-ckpt", default=None, help="skip training; rollout-eval this checkpoint")
    main(p.parse_args())
