"""Run ONCE on the held-out test set, after all tuning is done."""
import argparse, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from train import build_model, MEAN, STD

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--test_dir", required=True)
a = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(a.ckpt, map_location=device)
m = build_model(len(ck["classes"]), pretrained=False).to(device)
m.load_state_dict(ck["model"]); m.eval()

s = ck["img_size"]
tf = transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
ds = datasets.ImageFolder(a.test_dir, tf)
assert ds.classes == ck["classes"], "class order mismatch"

preds, labels = [], []
with torch.no_grad():
    for x, y in DataLoader(ds, batch_size=64):
        preds += m(x.to(device)).argmax(1).cpu().tolist(); labels += y.tolist()
print(classification_report(labels, preds, target_names=ds.classes, digits=3))
print(confusion_matrix(labels, preds))
