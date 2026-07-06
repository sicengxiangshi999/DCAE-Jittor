"""
train_jittor.py — DCAE training script rewritten for the Jittor framework.

Usage (single GPU):
    python train_jittor.py -d <dataset_path> -lr 1e-4 --cuda --epochs 50 \
        --lr_epoch 46 --batch-size 8 --save_path <checkpoint_dir> --save

Usage (load pretrained PyTorch checkpoint):
    python train_jittor.py -d <dataset_path> --checkpoint <pt_ckpt> \
        --save_path <checkpoint_dir> --cuda
"""

import os
import argparse
import math
import random
import sys
import time
import glob as _glob
from PIL import Image

import numpy as np
import jittor as jt
import jittor.nn as nn
from jittor import transform

from models_jittor import DCAE
from log_utils import save_log, save_recon_image

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
jt.flags.use_cuda = 0  # will be set by --cuda flag
# Disable operator fusion — DCAE produces fused ops > 4GB that OOM on 24GB GPU
jt.flags.lazy_execution = 0

# ---------------------------------------------------------------------------
# MS-SSIM implementation (pure Jittor)
# ---------------------------------------------------------------------------

def _fspecial_gauss(size, sigma):
    coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g).astype(np.float32)


def compute_ssim(img1, img2, K=(0.01, 0.03), win_size=11):
    C1, C2 = K[0] ** 2, K[1] ** 2
    kernel_np = _fspecial_gauss(win_size, 1.5)
    kernel = jt.array(kernel_np).unsqueeze(0).unsqueeze(0)

    def filt(img):
        c = img.shape[1]
        k = kernel.expand(c, 1, win_size, win_size)
        return nn.conv2d(img, k, padding=win_size // 2, groups=c)

    mu1, mu2 = filt(img1), filt(img2)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filt(img1 ** 2) - mu1_sq
    sigma2_sq = filt(img2 ** 2) - mu2_sq
    sigma12 = filt(img1 * img2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compute_ms_ssim(img1, img2, K=(0.01, 0.03), win_size=11):
    weights = jt.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.shape[0]
    mssim_list = []
    mcs_list = []
    C1, C2 = K[0] ** 2, K[1] ** 2

    kernel_np = _fspecial_gauss(win_size, 1.5)
    kernel = jt.array(kernel_np).unsqueeze(0).unsqueeze(0)

    for i in range(levels):
        c = img1.shape[1]
        k = kernel.expand(c, 1, win_size, win_size)
        mu1 = nn.conv2d(img1, k, padding=win_size // 2, groups=c)
        mu2 = nn.conv2d(img2, k, padding=win_size // 2, groups=c)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = nn.conv2d(img1 ** 2, k, padding=win_size // 2, groups=c) - mu1_sq
        sigma2_sq = nn.conv2d(img2 ** 2, k, padding=win_size // 2, groups=c) - mu2_sq
        sigma12 = nn.conv2d(img1 * img2, k, padding=win_size // 2, groups=c) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        mssim_list.append(ssim_map.mean())
        mcs_list.append(cs_map.mean())

        if i < levels - 1:
            img1 = nn.avg_pool2d(img1, 2, stride=2)
            img2 = nn.avg_pool2d(img2, 2, stride=2)

    # Product of CS^(weight) for all but last level, times SSIM^(weight) for last level
    mcs_arr = jt.concat([m.unsqueeze(0) for m in mcs_list[:-1]])
    # Use maximum with small epsilon to avoid numerical issues instead of relu
    mcs_pos = jt.maximum(mcs_arr, jt.array([1e-6]))
    ssim_pos = jt.maximum(mssim_list[-1], jt.array([1e-6]))
    prod = jt.prod(mcs_pos ** weights[:-1])
    ms_ssim_val = prod * ssim_pos ** weights[-1]
    return ms_ssim_val

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_compute_psnr(a, b):
    mse = jt.mean((a - b) ** 2).item()
    return -10 * math.log10(mse) if mse > 0 else float('inf')


def test_compute_msssim(a, b):
    return -10 * math.log10(1 - compute_ms_ssim(a, b).item())

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageDataset(jt.dataset.Dataset):
    """Simple image-folder dataset compatible with Jittor."""

    def __init__(self, root, split="train", patch_size=(256, 256)):
        super().__init__()
        self.patch_size = patch_size
        self.is_train = (split == "train")
        search = os.path.join(root, split, "**", "*.*")
        self.files = sorted([
            f for f in _glob.glob(search, recursive=True)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ])
        self.set_attrs(total_len=len(self.files))

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0

        if self.is_train:
            H, W = img.shape[:2]
            ph, pw = self.patch_size
            if H >= ph and W >= pw:
                y = random.randint(0, H - ph)
                x = random.randint(0, W - pw)
                img = img[y:y + ph, x:x + pw]
            else:
                img = np.pad(img, ((0, max(ph - H, 0)), (0, max(pw - W, 0)), (0, 0)))
                img = img[:ph, :pw]
        else:
            H, W = img.shape[:2]
            ph, pw = self.patch_size
            y = max((H - ph) // 2, 0)
            x = max((W - pw) // 2, 0)
            img = img[y:y + ph, x:x + pw]

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        return jt.array(img)

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class RateDistortionLoss(nn.Module):
    """Rate-distortion loss with Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, loss_type="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = loss_type

    def execute(self, output, target):
        N, _, H, W = target.shape
        num_pixels = N * H * W
        out = {}

        out["bpp_loss"] = sum(
            jt.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in output["likelihoods"].values()
        )

        if self.type == "mse":
            # Manual MSE calculation for numerical stability
            mse_val = jt.mean((output["x_hat"] - target) ** 2)
            out["mse_loss"] = mse_val
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out["ms_ssim_loss"] = compute_ms_ssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out["ms_ssim_loss"]) + out["bpp_loss"]

        return out

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, args):
    """Separate main parameters from entropy-bottleneck quantile parameters."""
    parameters = set()
    aux_parameters = set()
    for n, p in net.named_parameters():
        if n.endswith("._quantiles"):
            aux_parameters.add(n)
        else:
            parameters.add(n)

    params_dict = dict(net.named_parameters())
    assert len(parameters & aux_parameters) == 0, "Parameter overlap detected"
    assert len(parameters) > 0, f"No main parameters found. named_parameters returned: {list(params_dict.keys())[:10]}"

    optimizer = nn.Adam(
        [params_dict[n] for n in sorted(parameters)],
        lr=args.learning_rate,
    )
    # Manually add lr to param_groups for scheduler compatibility
    for pg in optimizer.param_groups:
        pg["lr"] = args.learning_rate

    aux_list = [params_dict[n] for n in sorted(aux_parameters)]
    if aux_list:
        aux_optimizer = nn.Adam(aux_list, lr=args.aux_learning_rate)
        for pg in aux_optimizer.param_groups:
            pg["lr"] = args.aux_learning_rate
    else:
        # Jittor Adam needs at least one param; use a dummy with zero grad
        dummy = nn.Parameter(jt.array([0.0]))
        aux_optimizer = nn.Adam([dummy], lr=0)
        for pg in aux_optimizer.param_groups:
            pg["lr"] = 0
    return optimizer, aux_optimizer


class MultiStepLR:
    """Simple multi-step learning rate scheduler."""

    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.milestones = set(milestones) if milestones else set()
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch in self.milestones:
            for pg in self.optimizer.param_groups:
                pg["lr"] *= self.gamma

    def get_last_lr(self):
        # Jittor optimizers may not have "lr" key; inspect param_groups structure
        try:
            return [pg["lr"] for pg in self.optimizer.param_groups]
        except KeyError:
            # Fallback: check if param_groups has 'initial_lr' or other lr-like keys
            lrs = []
            for pg in self.optimizer.param_groups:
                # Try common lr keys
                lr = pg.get("lr", pg.get("initial_lr", pg.get("base_lr", 0.0)))
                lrs.append(lr)
            return lrs

# ---------------------------------------------------------------------------
# Pad / crop helpers
# ---------------------------------------------------------------------------

def pad(x, p=64):
    h, w = x.shape[2], x.shape[3]
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    pl = (new_w - w) // 2
    pr = new_w - w - pl
    pt = (new_h - h) // 2
    pb = new_h - h - pt
    x_padded = nn.pad(x, (pl, pr, pt, pb))
    return x_padded, (pl, pr, pt, pb)


def crop(x, padding):
    pl, pr, pt, pb = padding
    return x[:, :, pt:x.shape[2] - pb, pl:x.shape[3] - pr]

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(state, is_best, epoch, save_path, filename):
    jt.save(state, os.path.join(save_path, "checkpoint_latest.pkl"))
    if epoch % 5 == 0:
        jt.save(state, filename)
    if is_best:
        jt.save(state, os.path.join(save_path, "checkpoint_best.pkl"))


def load_pytorch_checkpoint(ckpt_path, net, optimizer=None, aux_optimizer=None, lr_scheduler=None):
    """Load a PyTorch checkpoint into the Jittor model (for transfer / resume)."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    jittor_state = {}
    for k, v in state.items():
        jittor_state[k] = v.detach().cpu().numpy()

    net.load_state_dict(jittor_state, strict=False)
    last_epoch = 0

    if "epoch" in ckpt and optimizer is not None:
        last_epoch = ckpt["epoch"] + 1
        # Note: optimizer state transfer is non-trivial between frameworks;
        # here we only restore the epoch counter.
    return last_epoch

# ---------------------------------------------------------------------------
# Train / test
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, dataloader, optimizer, aux_optimizer,
                    epoch, clip_max_norm, loss_type, lr_scheduler=None,
                    logger=None, total_samples=0):
    model.train()
    now_time = time.time()

    # epoch 级别的累积统计
    epoch_loss = AverageMeter()
    epoch_bpp = AverageMeter()
    epoch_aux = AverageMeter()
    if loss_type == "mse":
        epoch_mse = AverageMeter()
    else:
        epoch_ms_ssim = AverageMeter()

    for i, d in enumerate(dataloader):
        if not isinstance(d, jt.Var):
            d = jt.array(np.array(d))

        # Model forward
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        loss = out_criterion["loss"]

        # Main backward + step
        optimizer.backward(loss)
        optimizer.step()

        # Aux loss（必须在清空 _last_data 之前调用）
        aux_loss = model.aux_loss()
        aux_optimizer.backward(aux_loss)
        aux_optimizer.step()

        # Accumulate stats BEFORE releasing tensors
        epoch_loss.update(out_criterion["loss"].item(), d.shape[0])
        epoch_bpp.update(out_criterion["bpp_loss"].item(), d.shape[0])
        epoch_aux.update(aux_loss.item(), d.shape[0])
        if loss_type == "mse":
            epoch_mse.update(out_criterion["mse_loss"].item(), d.shape[0])
        else:
            epoch_ms_ssim.update(out_criterion["ms_ssim_loss"].item(), d.shape[0])

        # Release intermediate references + sync
        model.entropy_bottleneck._last_data = None
        jt.sync()

        if (i + 1) % 100 == 0:
            pre_time = now_time
            now_time = time.time()
            lr_str = f"{lr_scheduler.get_last_lr()[0]:.2e}" if lr_scheduler and len(lr_scheduler.get_last_lr()) > 0 else "-"
            print(f"time : {now_time - pre_time:.1f}s")
            print(f"lr : {lr_str}")
            if loss_type == "mse":
                print(
                    f"Train epoch {epoch}: "
                    f"[{(i + 1) * d.shape[0]}/{total_samples}] "
                    f"\tLoss: {out_criterion['loss'].item():.3f} |"
                    f"\tMSE loss: {out_criterion['mse_loss'].item():.3f} |"
                    f"\tBpp loss: {out_criterion['bpp_loss'].item():.2f} |"
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: "
                    f"[{(i + 1) * d.shape[0]}/{total_samples}] "
                    f"\tLoss: {out_criterion['loss'].item():.3f} |"
                    f"\tMS_SSIM loss: {out_criterion['ms_ssim_loss'].item():.3f} |"
                    f"\tBpp loss: {out_criterion['bpp_loss'].item():.2f} |"
                    f"\tAux loss: {aux_loss.item():.2f}"
                )

    # epoch 结束时写入 JSON 日志
    if logger is not None:
        log_data = {
            "train_loss": epoch_loss.avg,
            "bpp": epoch_bpp.avg,
            "aux_loss": epoch_aux.avg,
            "lr": lr_scheduler.get_last_lr()[0] if lr_scheduler else 0,
        }
        if loss_type == "mse":
            log_data["mse_loss"] = epoch_mse.avg
        else:
            log_data["ms_ssim_loss"] = epoch_ms_ssim.avg
        save_log(logger, "jittor", epoch, **log_data)


def test_epoch(epoch, test_dataloader, model, criterion, loss_type="mse", logger=None):
    model.eval()
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr_meter = AverageMeter()

    if loss_type == "mse":
        mse_loss = AverageMeter()
    else:
        ms_ssim_loss = AverageMeter()

    for d in test_dataloader:
        if not isinstance(d, jt.Var):
            d = jt.array(np.array(d))
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        loss.update(out_criterion["loss"].item(), d.shape[0])
        aux_loss.update(model.aux_loss().item(), d.shape[0])
        bpp_loss.update(out_criterion["bpp_loss"].item(), d.shape[0])
        if loss_type == "mse":
            mse_loss.update(out_criterion["mse_loss"].item(), d.shape[0])
        else:
            ms_ssim_loss.update(out_criterion["ms_ssim_loss"].item(), d.shape[0])
        psnr_meter.update(test_compute_psnr(d, out_net["x_hat"].clamp(0, 1)))
        jt.sync()  # prevent test-time memory pileup

    if loss_type == "mse":
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tPSNR: {psnr_meter.avg:.2f} dB |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )
    else:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tPSNR: {psnr_meter.avg:.2f} dB |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    if logger is not None:
        log_data = {
            "test_loss": loss.avg,
            "psnr": psnr_meter.avg,
            "test_bpp": bpp_loss.avg,
            "test_aux": aux_loss.avg,
        }
        if loss_type == "mse":
            log_data["test_mse"] = mse_loss.avg
        else:
            log_data["test_ms_ssim"] = ms_ssim_loss.avg
        save_log(logger, "jittor", epoch, **log_data)

    return loss.avg

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv):
    parser = argparse.ArgumentParser(description="DCAE Jittor training script")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=8)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256),
                        help="Size of the patches to be cropped (default: %(default)s)")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_type", type=str, default="jittor",
                        choices=["jittor", "pytorch"],
                        help="Checkpoint format (pytorch for transfer learning)")
    parser.add_argument("--type", type=str, default="mse", choices=["mse", "ms-ssim"])
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--lr_epoch", nargs="+", type=int, default=None)
    parser.add_argument("--continue_train", action="store_true", default=True)
    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    if args.cuda:
        jt.flags.use_cuda = 1

    loss_type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    os.makedirs(save_path, exist_ok=True)

    # 跨框架对比日志目录
    log_dir = os.path.join(os.path.dirname(__file__) or ".", "compare_logs")
    os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)
    logger_path = os.path.join(log_dir, "jittor_log.json")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ---- tensorboard (optional) ----
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(save_path, "tensorboard/"))
    except ImportError:
        writer = None
        print("[WARN] tensorboard not available; install torch to enable logging.")

    # ---- datasets ----
    train_dataset = ImageDataset(args.dataset, split="train", patch_size=tuple(args.patch_size))
    test_dataset = ImageDataset(args.dataset, split="test", patch_size=tuple(args.patch_size))

    train_dataloader = jt.dataset.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    test_dataloader = jt.dataset.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ---- model ----
    net = DCAE()

    # ---- optimizers ----
    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch or []
    print(f"milestones: {milestones}")
    lr_scheduler = MultiStepLR(optimizer, milestones, gamma=0.1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, loss_type=loss_type)

    # ---- checkpoint ----
    last_epoch = 0
    if args.checkpoint:
        print(f"Loading {args.checkpoint}")
        if args.checkpoint_type == "pytorch":
            last_epoch = load_pytorch_checkpoint(
                args.checkpoint, net, optimizer, aux_optimizer, lr_scheduler,
            )
        else:
            ckpt = jt.load(args.checkpoint)
            net.load_state_dict(ckpt["state_dict"])
            if args.continue_train:
                last_epoch = ckpt.get("epoch", 0) + 1
                # Optimizer states are Jittor arrays, directly loadable
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if "aux_optimizer" in ckpt:
                    aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
                if "lr_scheduler_epoch" in ckpt:
                    lr_scheduler.epoch = ckpt["lr_scheduler_epoch"]

    # ---- training loop ----
    best_loss = float("inf")
    total_samples = train_dataset.total_len
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            net, criterion, train_dataloader, optimizer, aux_optimizer,
            epoch, args.clip_max_norm, loss_type, lr_scheduler,
            logger=logger_path, total_samples=total_samples,
        )

        # Test phase
        jt.sync()
        loss = test_epoch(epoch, test_dataloader, net, criterion, loss_type, logger=logger_path)
        jt.sync()
        if writer is not None:
            writer.add_scalar("test_loss", loss, epoch)

        # 每 5 个 epoch 保存一次重建图片对比
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            net.eval()
            for sample_batch in test_dataloader:
                if not isinstance(sample_batch, jt.Var):
                    sample_batch = jt.array(np.array(sample_batch))
                sample_out = net(sample_batch)
                save_recon_image(
                    sample_batch, sample_out["x_hat"],
                    os.path.join(log_dir, "samples"),
                    epoch, "jittor", max_images=2,
                )
                break
            jt.sync()  # sync after reconstruction

        # Update learning rate scheduler AFTER train_one_epoch
        if lr_scheduler:
            lr_scheduler.step()
            # Debug print learning rates
            lrs = lr_scheduler.get_last_lr()
            print(f"[DEBUG] Epoch {epoch} LR: {[f'{lr:.2e}' for lr in lrs]}")

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler_epoch": lr_scheduler.epoch,
                },
                is_best,
                epoch,
                save_path,
                os.path.join(save_path, f"{epoch}_checkpoint.pkl"),
            )


if __name__ == "__main__":
    main(sys.argv[1:])
