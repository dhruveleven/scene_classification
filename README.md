# Scene Classification (Intel Image Classification)

Fine-tune a pretrained ResNet18 on 6 scene classes (buildings, forest, glacier, mountain, sea, street).
Workflow: **develop on laptop (CPU) > train on Kaggle (GPU) > run ONNX inference on laptop (CPU)**.

## Files
| File | Purpose |
|---|---|
| `train.py` | Training: frozen-backbone warmup, AMP, cosine LR, checkpoint + resume, metrics.csv |
| `evaluate.py` | Held-out test set report + confusion matrix (run once at the end) |
| `export_onnx.py` | `best.pt` > `model.onnx` + `model.json` (classes, img size) |
| `predict.py` | CPU inference with ONNX Runtime, prints latency |
| `configs/local.yaml` | Laptop smoke-test settings |
| `configs/kaggle.yaml` | Full GPU run settings |
| `kaggle_train.ipynb` | Thin runner notebook to import into Kaggle |

## Phase 1: Local (Windows, PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Kaggle API token: kaggle.com > Settings > API > Create New Token
# Put kaggle.json in C:\Users\<you>\.kaggle\
kaggle datasets download -d puneet6060/intel-image-classification -p data --unzip

# Smoke test: proves the pipeline runs end to end (~1-2 min on CPU)
python train.py --config configs/local.yaml --limit 200

# Overfit check: train loss should drop near the floor. If it can't overfit 32 images, there is a bug.
python train.py --config configs/local.yaml --limit 32 --epochs 30 --freeze_epochs 0 --out_dir runs/overfit
```
Then push to GitHub (`data/`, `runs/`, weights are gitignored).

## Phase 2: Kaggle (GPU)
1. kaggle.com > Code > New Notebook > File > Import Notebook > `kaggle_train.ipynb`.
2. Settings: Accelerator **GPU T4**, Internet **On** (needs phone verification). Add Input: **Intel Image Classification**.
3. Set your repo URL in cell 3, run cells 1 to 4 interactively.
4. **Save Version > Save & Run All (Commit)** for the full run. Download `best.pt` from the Output tab.

## Phase 3: Back on the laptop
```powershell
python export_onnx.py --ckpt best.pt
python predict.py model.onnx data\seg_pred\seg_pred\<any>.jpg
```

## Results
| Run | Img size | Epochs | Val acc | Test acc | CPU latency |
|---|---|---|---|---|---|
| exp1 | 224 | 8 | | | |
