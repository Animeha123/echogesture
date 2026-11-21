"""Live HUD demo: camera -> on-device landmarks -> INT8 classifier -> captions (+ optional speech).

    python scripts/live_demo.py                  # default webcam
    python scripts/live_demo.py --source clip.mp4 --record demo.mp4
    python scripts/live_demo.py --speak          # offline TTS via pyttsx3

Keys: q quit | c clear captions | s toggle speech
Sign segmentation: a sign is committed when the hands drop out of view (or rest)
after a signing segment; while signing, a live guess is shown every few frames.
"""
import argparse
import collections
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import config as C  # noqa: E402
from echogesture.features import compute_features  # noqa: E402
from echogesture.landmarks import LandmarkExtractor  # noqa: E402
from echogesture.runtime import Classifier  # noqa: E402

CYAN, PINK, WHITE, GREY = (235, 220, 40), (170, 40, 230), (245, 245, 245), (150, 150, 150)


class Speaker:
    def __init__(self):
        self.q, self.ok = queue.Queue(), True
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            self.ok = False
            print("pyttsx3 not installed -> speech disabled (pip install pyttsx3)")
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import pyttsx3
        eng = pyttsx3.init()
        while True:
            eng.say(self.q.get())
            eng.runAndWait()

    def say(self, text):
        if self.ok:
            self.q.put(text)


def panel(img, x0, y0, x1, y1, alpha=0.55):
    sub = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(sub, 1 - alpha, np.full_like(sub, (40, 20, 15)), alpha, 0)


def draw_hud(img, lm, aspect, state):
    h, w = img.shape[:2]
    # hand landmarks
    for o, col in ((C.LH0, CYAN), (C.RH0, PINK)):
        pts = lm[o:o + C.N_HAND]
        if np.isnan(pts[0, 0]):
            continue
        for x, y, _ in pts:
            cv2.circle(img, (int(x / aspect * w), int(y * h)), 3, col, -1, cv2.LINE_AA)
    # top status bar
    panel(img, 0, 0, w, 44)
    cv2.putText(img, "EchoGesture", (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, CYAN, 1, cv2.LINE_AA)
    badge = "ON-DEVICE  |  OFFLINE"
    cv2.putText(img, badge, (190, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 230, 120), 1, cv2.LINE_AA)
    info = f"{state['fps']:4.1f} fps   lat {state['lat']:5.1f} ms   {'SIGNING' if state['signing'] else 'idle'}"
    cv2.putText(img, info, (w - 360, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    # live guess (top-3 with bars)
    if state["guess"]:
        panel(img, w - 250, 56, w - 10, 56 + 30 * len(state["guess"]) + 12)
        for i, (g, p) in enumerate(state["guess"]):
            y = 80 + 30 * i
            cv2.rectangle(img, (w - 240, y - 14), (w - 240 + int(150 * p), y + 4), PINK if i == 0 else GREY, -1)
            cv2.putText(img, f"{g} {p * 100:.0f}%", (w - 236, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    # captions
    panel(img, 0, h - 80, w, h, 0.7)
    text = " ".join(state["captions"][-6:]) or "(sign to start)"
    scale = 1.2 if len(text) < 40 else 0.9
    cv2.putText(img, text, (16, h - 30), cv2.FONT_HERSHEY_DUPLEX, scale, WHITE, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--mirror", action="store_true", help="mirror the display (selfie view)")
    ap.add_argument("--record", default="")
    a = ap.parse_args()

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"cannot open source {a.source}")
    ext, clf = LandmarkExtractor(), Classifier()
    speaker = Speaker() if a.speak else None
    writer = None

    buf = collections.deque(maxlen=C.MAX_VIDEO_FRAMES)
    state = dict(fps=0.0, lat=0.0, signing=False, guess=[], captions=[])
    absent, n_seg, frame_i, t_prev = 0, 0, 0, time.perf_counter()
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        ts = int((t0 - t_start) * 1000)
        lm = ext.process_frame(frame, ts)
        frame_ms = (time.perf_counter() - t0) * 1000
        hands = not (np.isnan(lm[C.LH0, 0]) and np.isnan(lm[C.RH0, 0]))

        if hands:
            absent = 0
            state["signing"] = True
            buf.append(lm)
            n_seg += 1
            if n_seg >= 12 and frame_i % 6 == 0:          # live guess
                t1 = time.perf_counter()
                probs = clf_probs(clf, np.stack(buf))
                state["lat"] = frame_ms + (time.perf_counter() - t1) * 1000
                top = np.argsort(-probs)[:3]
                state["guess"] = [(clf.labels[i], float(probs[i])) for i in top]
        else:
            absent += 1
            if state["signing"]:
                buf.append(lm)
            if state["signing"] and absent >= 6:          # sign finished -> commit
                if n_seg >= 10:
                    probs = clf_probs(clf, np.stack(buf))
                    i = int(np.argmax(probs))
                    word = clf.labels[i]
                    if probs[i] >= a.threshold and (not state["captions"] or state["captions"][-1] != word):
                        state["captions"].append(word)
                        if speaker:
                            speaker.say(word.lower())
                buf.clear()
                n_seg = 0
                state["signing"] = False
                state["guess"] = []

        now = time.perf_counter()
        state["fps"] = 0.9 * state["fps"] + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        frame_i += 1

        disp = cv2.flip(frame, 1) if a.mirror else frame.copy()
        lm_draw = lm.copy()
        aspect = frame.shape[1] / frame.shape[0]
        if a.mirror:
            lm_draw[:, 0] = aspect - lm_draw[:, 0]
        draw_hud(disp, lm_draw, aspect, state)
        if a.record:
            if writer is None:
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 25,
                                         (disp.shape[1], disp.shape[0]))
            writer.write(disp)
        cv2.imshow("EchoGesture", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            state["captions"].clear()
        if k == ord("s"):
            speaker = None if speaker else Speaker()

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def clf_probs(clf, seq):
    from echogesture.runtime import softmax
    return softmax(clf.logits(compute_features(seq)))[0]


if __name__ == "__main__":
    main()
