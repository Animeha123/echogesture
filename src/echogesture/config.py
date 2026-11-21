"""Shared constants. Training and inference both import from here so the
feature pipeline can never drift between the two."""
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = SRC_DIR / "models"
DATA_DIR = SRC_DIR.parent / "data"            # raw videos + cached landmarks (not shipped)
RESULTS_DIR = SRC_DIR.parent / "results"

# MediaPipe task bundles (downloaded once by scripts/download_mediapipe_models.py)
HAND_TASK = MODELS_DIR / "hand_landmarker.task"
POSE_TASK = MODELS_DIR / "pose_landmarker_lite.task"
HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/latest/hand_landmarker.task")
POSE_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

# Our classifier
ONNX_FP32 = MODELS_DIR / "echogesture_fp32.onnx"
ONNX_INT8 = MODELS_DIR / "echogesture_int8.onnx"
LABELS_TXT = MODELS_DIR / "labels.txt"
CHECKPOINT = MODELS_DIR / "echogesture.pt"

# Landmark layout stored per frame: (N_POINTS, 3)
#   0..10  -> upper-body pose points (see POSE_IDX)
#   11..31 -> left hand (21)
#   32..52 -> right hand (21)
POSE_IDX = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]  # nose, eyes, mouth, shoulders, elbows, wrists
N_POSE = len(POSE_IDX)
N_HAND = 21
LH0, RH0 = N_POSE, N_POSE + N_HAND
N_POINTS = N_POSE + 2 * N_HAND               # 53
# indices inside our 11-point pose subset
P_LSHO, P_RSHO, P_LWRI, P_RWRI = 5, 6, 9, 10

# Temporal window fed to the classifier
SEQ_LEN = 48
MAX_VIDEO_FRAMES = 160      # frames sampled per clip before trimming (speed cap)
FEAT_DIM = 352              # asserted in features.py

# Processing resolution: frames are downscaled so the long side is <= this
# (384 is ~30% faster than 480 for MediaPipe; override with env ECHO_PROC_SIDE=480)
import os as _os
PROC_LONG_SIDE = int(_os.environ.get("ECHO_PROC_SIDE", "384"))
# Run the pose model every N frames (body moves slowly; hands run every frame)
POSE_EVERY = 2
