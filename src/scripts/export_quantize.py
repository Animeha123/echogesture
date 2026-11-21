"""Conversion script: PyTorch checkpoint -> ONNX FP32 -> ONNX INT8 (static, QDQ, per-channel).

    python scripts/export_quantize.py
    python scripts/export_quantize.py --tflite      # optional extra TFLite INT8 via onnx2tf

Calibration uses real (clean) training features so activation ranges match
deployment. Prints FP32 vs INT8 accuracy on the val split and file sizes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from echogesture import config as C  # noqa: E402
from echogesture.features import compute_features  # noqa: E402
from echogesture.model import build_model  # noqa: E402


def export_onnx(model, path: Path):
    x = torch.randn(1, C.SEQ_LEN, C.FEAT_DIM)
    kw = dict(input_names=["features"], output_names=["logits"], opset_version=17,
              dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, (x,), str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, (x,), str(path), **kw)
    except Exception as e:  # legacy exporter removed in newer torch -> dynamo exporter
        print(f"legacy export failed ({e}); trying dynamo exporter")
        torch.onnx.export(model, (x,), str(path), dynamo=True, input_names=["features"],
                          output_names=["logits"], opset_version=18,
                          dynamic_shapes={"x": {0: torch.export.Dim("batch", max=256)}})
    import onnx
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    # ensure weights are embedded in a single file
    onnx.save_model(m, str(path), save_as_external_data=False)


class CalibReader:
    def __init__(self, feats: np.ndarray, name: str):
        self.it = iter([{name: f[None]} for f in feats])

    def get_next(self):
        return next(self.it, None)


def accuracy(sess, feats, y):
    name = sess.get_inputs()[0].name
    logits = np.concatenate([sess.run(None, {name: feats[i:i + 64]})[0] for i in range(0, len(feats), 64)])
    top = np.argsort(-logits, 1)[:, :5]
    return (top[:, 0] == y).mean(), (top == y[:, None]).any(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=C.CHECKPOINT)
    ap.add_argument("--index", type=Path, default=C.DATA_DIR / "wlasl_index.csv")
    ap.add_argument("--landmarks", type=Path, default=C.DATA_DIR / "landmarks")
    ap.add_argument("--n-calib", type=int, default=512)
    ap.add_argument("--tflite", action="store_true")
    a = ap.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from train import load_split

    ck = torch.load(a.ckpt, map_location="cpu")
    model = build_model(ck["cfg"]).eval()
    model.load_state_dict(ck["state_dict"])
    # Class names come from the index (label int -> gloss). Older checkpoints stored labels
    # split on whitespace ("THANK YOU" -> 2 entries), which shifted every later name.
    idx = C.DATA_DIR / "wlasl_index.csv"
    if idx.exists():
        import csv
        names = {}
        with open(idx, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                names[int(r["label"])] = r["gloss"].upper()
        labels = [names[i] for i in range(max(names) + 1)]
    else:
        labels = ck["labels"]
    C.LABELS_TXT.write_text("\n".join(labels) + "\n", encoding="utf-8")

    export_onnx(model, C.ONNX_FP32)
    print(f"FP32 ONNX -> {C.ONNX_FP32} ({C.ONNX_FP32.stat().st_size / 1e6:.2f} MB)")

    Xtr, _ = load_split(a.index, a.landmarks, {"train"})
    rng = np.random.default_rng(0)
    pick = rng.choice(len(Xtr), size=min(a.n_calib, len(Xtr)), replace=False)
    calib = np.stack([compute_features(Xtr[i]) for i in pick])

    pre = C.MODELS_DIR / "_pre.onnx"
    quant_pre_process(str(C.ONNX_FP32), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(C.ONNX_INT8), CalibReader(calib, "features"),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"ActivationSymmetric": True, "CalibPercentile": 99.99})
    pre.unlink(missing_ok=True)
    print(f"INT8 ONNX -> {C.ONNX_INT8} ({C.ONNX_INT8.stat().st_size / 1e6:.2f} MB)")

    Xva, yva = load_split(a.index, a.landmarks, {"val"})
    if len(Xva):
        fv = np.stack([compute_features(x) for x in Xva])
        so = ort.SessionOptions()
        for p in (C.ONNX_FP32, C.ONNX_INT8):
            s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
            t1, t5 = accuracy(s, fv, yva)
            print(f"  {p.name:24s} val top1 {t1 * 100:5.2f}%  top5 {t5 * 100:5.2f}%")

    if a.tflite:
        try:
            import subprocess
            out_dir = C.MODELS_DIR / "tflite"
            np.save(C.MODELS_DIR / "_calib.npy", calib)
            subprocess.run([sys.executable, "-m", "onnx2tf", "-i", str(C.ONNX_FP32), "-o", str(out_dir),
                            "-oiqt", "-cind", "features", str(C.MODELS_DIR / "_calib.npy"),
                            "[[[0.0]]]", "[[[1.0]]]"], check=True)
            print(f"TFLite models -> {out_dir}")
        except Exception as e:
            print(f"TFLite conversion skipped: {e}  (pip install onnx2tf tensorflow)")


if __name__ == "__main__":
    main()
