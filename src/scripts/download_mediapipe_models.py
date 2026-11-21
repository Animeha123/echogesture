"""One-time setup: fetch the MediaPipe hand + pose .task bundles into src/models/.

This is the only step that touches the network; inference afterwards is fully offline.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import config as C  # noqa: E402


def main():
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for url, dst in [(C.HAND_TASK_URL, C.HAND_TASK), (C.POSE_TASK_URL, C.POSE_TASK)]:
        if dst.exists() and dst.stat().st_size > 0:
            print(f"ok   {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"get  {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"ok   {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
