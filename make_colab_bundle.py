"""Pack what Colab needs for GPU training into colab_bundle.zip (~25-30 MB for 100 words).

Training only needs the cached landmarks + index + code - no videos, no MediaPipe.
Run AFTER extract_landmarks.py:

    python make_colab_bundle.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    lm = sorted((ROOT / "data" / "landmarks").glob("*.npy"))
    if not lm:
        raise SystemExit("No landmarks in data/landmarks - run scripts/extract_landmarks.py first.")
    for need in ["data/wlasl_index.csv", "src/models/labels.txt"]:
        if not (ROOT / need).exists():
            raise SystemExit(f"Missing {need} - run scripts/prepare_wlasl.py first.")

    out = ROOT / "colab_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in list((ROOT / "src" / "echogesture").glob("*.py")) + list((ROOT / "src" / "scripts").glob("*.py")):
            z.write(p, p.relative_to(ROOT))
        z.write(ROOT / "src" / "models" / "labels.txt", "src/models/labels.txt")
        z.write(ROOT / "data" / "wlasl_index.csv", "data/wlasl_index.csv")
        if (ROOT / "data" / "aslc_index.csv").exists():
            z.write(ROOT / "data" / "aslc_index.csv", "data/aslc_index.csv")
        pix = sorted((ROOT / "data" / "landmarks_pix").glob("*.npy"))
        for p in lm + pix:
            z.write(p, p.relative_to(ROOT))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(lm)} landmark files, "
          f"{len(pix)} pixel-perturbed variants)")
    print("Upload it in the Colab notebook (colab_train.ipynb).")


if __name__ == "__main__":
    main()
