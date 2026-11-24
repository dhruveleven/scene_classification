"""CPU inference with ONNX Runtime. No torch needed.
Usage: python predict.py model.onnx path/to/image.jpg [more images...]"""
import json, sys, time
import numpy as np, onnxruntime as ort
from PIL import Image

model_path, img_paths = sys.argv[1], sys.argv[2:]
meta = json.load(open(model_path.replace(".onnx", ".json")))
s = meta["img_size"]
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

for p_img in img_paths:
    img = Image.open(p_img).convert("RGB").resize((s, s), Image.BILINEAR)
    x = ((np.asarray(img, dtype=np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
    t = time.perf_counter()
    logits = sess.run(None, {"input": x})[0][0]
    ms = (time.perf_counter() - t) * 1000
    p = np.exp(logits - logits.max()); p /= p.sum()
    print(f"{p_img}: {meta['classes'][p.argmax()]} ({p.max():.2%}) in {ms:.1f} ms")
