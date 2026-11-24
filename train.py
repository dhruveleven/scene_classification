import argparse, csv, os, random, time
import numpy as np, torch, torch.nn as nn, yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_loaders(cfg, limit=None):
    sz = cfg["img_size"]
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(sz, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tf_eval = transforms.Compose([
        transforms.Resize((sz, sz)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    full_tr = datasets.ImageFolder(cfg["train_dir"], tf_train)
    full_ev = datasets.ImageFolder(cfg["train_dir"], tf_eval)  # same files, no augmentation
    idx = torch.randperm(len(full_tr), generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
    if limit:
        idx = idx[:limit]
    n_val = max(1, int(len(idx) * cfg["val_frac"]))
    tr, va = Subset(full_tr, idx[n_val:]), Subset(full_ev, idx[:n_val])
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              pin_memory=torch.cuda.is_available())
    return DataLoader(tr, shuffle=True, **kw), DataLoader(va, shuffle=False, **kw), full_tr.classes


def build_model(n_classes, pretrained=True):
    w = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def set_backbone_trainable(m, flag):
    for name, p in m.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = flag


@torch.no_grad()
def evaluate(m, loader, device):
    m.eval(); crit = nn.CrossEntropyLoss()
    correct = total = 0; loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = m(x)
        loss_sum += crit(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="use only N images (smoke test)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_dir", default=None, help="override config train_dir")
    ap.add_argument("--out_dir", default=None, help="override config out_dir")
    ap.add_argument("--freeze_epochs", type=int, default=None)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    for k in ("epochs", "train_dir", "out_dir", "freeze_epochs"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.no_pretrained:
        cfg["pretrained"] = False
    seed_all(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config.yaml"), "w") as f:  # record the exact run settings
        yaml.safe_dump(cfg, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    train_dl, val_dl, classes = make_loaders(cfg, a.limit)
    print(f"device {device} | classes {classes} | train {len(train_dl.dataset)} val {len(val_dl.dataset)}")

    model = build_model(len(classes), cfg.get("pretrained", True)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    start, best = 0, 0.0
    last_path = os.path.join(cfg["out_dir"], "last.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {start}")

    metrics_path = os.path.join(cfg["out_dir"], "metrics.csv")
    new_file = not os.path.exists(metrics_path)
    log = open(metrics_path, "a", newline="")
    writer = csv.writer(log)
    if new_file:
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "seconds"])

    for ep in range(start, cfg["epochs"]):
        set_backbone_trainable(model, ep >= cfg["freeze_epochs"])  # head-only warmup, then full fine-tune
        model.train(); run_loss = 0.0; n = 0; t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item() * y.size(0); n += y.size(0)
        sched.step()

        val_loss, val_acc = evaluate(model, val_dl, device)
        secs = time.time() - t0
        print(f"ep {ep} | train_loss {run_loss/n:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | {secs:.0f}s")
        writer.writerow([ep, round(run_loss / n, 5), round(val_loss, 5), round(val_acc, 5), round(secs, 1)]); log.flush()

        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "classes": classes, "img_size": cfg["img_size"]},
                       os.path.join(cfg["out_dir"], "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best}, last_path)
    print(f"best val_acc {best:.4f}")


if __name__ == "__main__":
    main()
