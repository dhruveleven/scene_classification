import argparse, json, torch
from train import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="model.onnx")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
m = build_model(len(ck["classes"]), pretrained=False)
m.load_state_dict(ck["model"]); m.eval()
s = ck["img_size"]
torch.onnx.export(m, torch.randn(1, 3, s, s), a.out, input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
json.dump({"classes": ck["classes"], "img_size": s}, open(a.out.replace(".onnx", ".json"), "w"))
print("exported", a.out)
