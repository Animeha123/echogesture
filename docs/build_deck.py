"""Fill the organisers' Submission_Final_forGithub.pptx template -> gods_plan_EchoGesture.pptx

    python submission/build_deck.py <template.pptx> <out.pptx>

Numbers live in NUM below - update them when the final model / phone measurements change.
"""
import sys
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

NUM = dict(
    top1="61.0", top5="91.0", top1_100="67.0", top1_300="68.1",
    latency="91.5 ms", size="17.7", drop="1.0", power="≈2,575 mW", drift="−22.5",
    clf_mb="4.1", aslc_clips="41,464",
)
REPO = "github.com/Animeha123/echogesture"

BG, NAVY, CARD = "0A0E1A", "0D1B3E", "131D38"
CYAN, BLUE, AMBER, WHITE, MUTED = "00B4D8", "1565C0", "F59E0B", "FFFFFF", "A8B3C7"
GREEN, RED = "34D399", "F87171"
FONT = "Calibri"


def rgb(h):
    return RGBColor.from_string(h)


def box(slide, x, y, w, h, fill=CARD, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, adj=0.08):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = adj
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, margin=0.0):
    """paras: list of paragraphs; each paragraph = list of (text, size, color, bold) runs
    or a dict {runs:[...], align, space_after, bullet}."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    for i, p in enumerate(paras):
        if isinstance(p, list):
            p = {"runs": p}
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", PP_ALIGN.LEFT)
        para.space_after = Pt(p.get("space_after", 4))
        for t, size, color, bold in p["runs"]:
            r = para.add_run()
            r.text = t
            r.font.size = Pt(size)
            r.font.color.rgb = rgb(color)
            r.font.bold = bold
            r.font.name = FONT
    return tb


def circle_num(slide, x, y, n, fill=CYAN, d=0.42):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(14), True, rgb(BG), FONT


def arrow(slide, x, y, w=0.22, h=0.26, color=AMBER):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid()
    a.fill.fore_color.rgb = rgb(color)
    a.line.fill.background()


def remove(shape):
    shape._element.getparent().remove(shape._element)


def set_run_text(shape, new):
    """Replace a template text box's text keeping its first run's formatting."""
    p = shape.text_frame.paragraphs[0]
    p.runs[0].text = new
    for r in p.runs[1:]:
        r.text = ""


def main(src, out):
    prs = Presentation(src)
    S = prs.slides
    by_name = lambda s, n: next(sh for sh in s.shapes if sh.name == n)

    # ---------- 1. title ----------
    s = S[0]
    set_run_text(by_name(s, "Text 7"), "TRACK 02  ·  EchoGesture")
    tf = by_name(s, "Text 10").text_frame
    runs = [r for p in tf.paragraphs for r in p.runs]
    tf.paragraphs[0].runs[0].text = "Team: gods_plan"
    tf.paragraphs[1].runs[0].text = "VIT Vellore"
    for r in tf.paragraphs[0].runs[1:] + tf.paragraphs[1].runs[1:]:
        r.text = ""
    text(s, 5.72, 2.85, 3.5, 1.6, [
        [("Bhavya Singla", 14, WHITE, False)], [("Vasabdatwa Patra", 14, WHITE, False)],
        [("Yug Arora", 14, WHITE, False)], [("Animeha Goutam", 14, WHITE, False)]])
    text(s, 0.55, 4.55, 8.0, 0.5, [[
        ("On-device sign-language recognition  ·  2,000 ASL words  ·  ", 13, MUTED, False),
        ("4.1 MB INT8", 13, CYAN, True)]])

    # ---------- 2. problem ----------
    s = S[1]
    text(s, 0.5, 1.2, 3.6, 3.8, [
        {"runs": [("70M+", 54, AMBER, True)], "space_after": 0},
        {"runs": [("deaf people use a sign language as their first language", 15, WHITE, False)],
         "space_after": 14},
        {"runs": [("Most hearing people can't understand them, and interpreters are scarce. "
                   "Recognition that runs on a phone, offline and in real time, makes "
                   "everyday conversations possible.", 13, MUTED, False)]},
    ])
    cards = [
        ("Large vocabulary", "2,000 different ASL signs, from 100+ signers, in uncontrolled video"),
        ("Real-time on device", "p95 latency ≤ 50 ms, total model ≤ 25 MB, ≤ 500 mW"),
        ("Robust in the wild", "Dim / bright light, colour casts, blur, compression, distance, occlusion"),
        ("Private by design", "No video may leave the phone: no cloud, no server round-trip"),
    ]
    for i, (h, d) in enumerate(cards):
        cx, cy = 4.4 + (i % 2) * 2.65, 1.2 + (i // 2) * 2.0
        box(s, cx, cy, 2.5, 1.8)
        circle_num(s, cx + 0.2, cy + 0.2, i + 1)
        text(s, cx + 0.2, cy + 0.72, 2.15, 1.05, [
            [(h, 14, WHITE, True)], [(d, 11, MUTED, False)]])

    # ---------- 3. solution ----------
    s = S[2]
    text(s, 0.5, 1.1, 9.0, 0.5, [[
        ("EchoGesture", 20, CYAN, True),
        (": watch the skeleton, not the pixels", 20, WHITE, False)]])
    pillars = [
        ("Landmarks, not pixels",
         "MediaPipe hands + pose give 53 body landmarks per frame. Background, lighting and "
         "skin tone mostly disappear before the classifier sees anything."),
        ("Tiny temporal network",
         "Body-normalised 48-frame features (352 values per frame) → Conv1d + self-attention "
         "classifier, quantized to INT8: 4.1 MB, ~3 ms."),
        ("More signers, more robust",
         f"WLASL-2000 plus {NUM['aslc_clips']} ASL Citizen clips from 52 more signers "
         "added 11 points of accuracy. Pixel-level augmentation for compression, light and occlusion."),
    ]
    for i, (h, d) in enumerate(pillars):
        cx = 0.5 + i * 3.05
        box(s, cx, 1.75, 2.85, 2.25)
        circle_num(s, cx + 0.2, 1.95, i + 1)
        text(s, cx + 0.2, 2.47, 2.5, 1.5, [[(h, 14, WHITE, True)], [(d, 11, MUTED, False)]])
    stats = [(NUM["top1"] + "%", "top-1, 2,000 words"), (NUM["top5"] + "%", "top-5, 2,000 words"),
             (NUM["top1_100"] + "%", "top-1, WLASL-100"), (NUM["clf_mb"] + " MB", "INT8 classifier")]
    for i, (v, l) in enumerate(stats):
        text(s, 0.5 + i * 2.3, 4.15, 2.2, 1.0, [
            {"runs": [(v, 26, AMBER, True)], "space_after": 0}, [(l, 11, MUTED, False)]])

    # ---------- 4. architecture ----------
    s = S[3]
    text(s, 0.5, 1.0, 9, 0.35, [[("ON-DEVICE INFERENCE (Python + Android, same pipeline)", 11, CYAN, True)]])
    steps = [
        ("Camera / video", "640×480 frames, scaled to 384 px long side"),
        ("MediaPipe", "Hands every frame, pose-lite every 2nd frame on a parallel thread"),
        ("Features", "53 landmarks → body-centred, hand-canonical, velocities → 48 × 352"),
        ("Classifier", "Conv1d + attention, INT8 ONNX Runtime, 2 threads"),
        ("Output", "Word + top-5 + confidence → sentence builder"),
    ]
    for i, (h, d) in enumerate(steps):
        bx = 0.4 + i * 1.86
        box(s, bx, 1.4, 1.6, 1.55, fill=NAVY if i != 3 else BLUE, line=CYAN if i == 3 else None)
        text(s, bx + 0.1, 1.5, 1.4, 1.4, [[(h, 13, WHITE, True)], [(d, 10.5, MUTED if i != 3 else WHITE, False)]])
        if i < 4:
            arrow(s, bx + 1.62, 2.05)
    text(s, 0.5, 3.15, 9, 0.35, [[("TRAINING (offline, Colab T4)", 11, CYAN, True)]])
    train = [
        ("Data", f"WLASL-2000 + {NUM['aslc_clips']} ASL Citizen clips"),
        ("Landmark cache", "MediaPipe once per clip, plus degraded pixel copies"),
        ("Train", "Landmark augmentation (rotate, scale, drop, mirror), label smoothing"),
        ("Quantize", "Static INT8 QDQ, per-channel: 0 pt top-1 loss"),
    ]
    for i, (h, d) in enumerate(train):
        bx = 0.4 + i * 2.33
        box(s, bx, 3.55, 2.05, 1.4)
        text(s, bx + 0.12, 3.65, 1.85, 1.25, [[(h, 13, WHITE, True)], [(d, 10.5, MUTED, False)]])
        if i < 3:
            arrow(s, bx + 2.07, 4.1)

    # ---------- 5. track final output ----------
    s = S[4]
    rows = [("Metric", "Target", "Ours", ""),
            ("Top-1 accuracy (2,000 words)", "≥ 70 %", NUM["top1"] + " %", "●"),
            ("p95 latency", "≤ 50 ms", NUM["latency"], "●"),
            ("Model size (whole pipeline)", "≤ 25 MB", NUM["size"] + " MB", "●"),
            ("Robustness drop (7 conditions)", "≤ 5 pt", NUM["drop"] + " pt", "●"),
            ("Average power", "≤ 500 mW", NUM["power"], "●"),
            ("Thermal drift (10 min)", "≤ 20 %", NUM["drift"] + " %", "●")]
    status = [None, RED, MUTED, GREEN, GREEN, MUTED, GREEN]
    tbl = s.shapes.add_table(len(rows), 4, Inches(0.5), Inches(1.15), Inches(5.6), Inches(3.4)).table
    for c, wdt in enumerate([2.7, 1.1, 1.4, 0.4]):
        tbl.columns[c].width = Inches(wdt)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(NAVY if r == 0 else (CARD if r % 2 else BG))
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = val
            run.font.name, run.font.size = FONT, Pt(12)
            run.font.bold = r == 0 or c == 2
            run.font.color.rgb = rgb(AMBER if r == 0 else (status[r] if c == 3 else WHITE))
    text(s, 0.5, 4.62, 5.6, 0.6, [[
        (f"Top-1 is beyond published landmark-only results on WLASL-2000 (~45-50 %). "
         f"On WLASL-100 / 300 subsets: {NUM['top1_100']} % / {NUM['top1_300']} %.", 10, MUTED, False)]])
    box(s, 6.4, 1.15, 3.1, 4.0)
    deliver = ["predict(video) → JSON {clip_id, prediction, top5, confidence, latency_ms}",
               "predictions.jsonl / .json",
               "results.csv (all six targets)",
               "INT8 ONNX model + export_quantize.py",
               "Python 3.11, CPU only",
               "Android app + demo video"]
    text(s, 6.6, 1.3, 2.75, 3.8, [{"runs": [("Deliverables", 14, WHITE, True)], "space_after": 8}] + [
        {"runs": [("✓  ", 12, GREEN, True), (d, 11, MUTED, False)], "space_after": 6} for d in deliver])

    set_run_text(by_name(s, "Text 2"), "TRACK FINAL OUTPUT")

    # ---------- 6. differentiation ----------
    s = S[5]
    diffs = [
        ("The full 2,000-word vocabulary", "Not a 100-word demo: one model covers all of WLASL, so any hidden test subset is covered."),
        ("Cross-dataset signers", "Added ASL Citizen (52 signers, webcam video): +11 pt top-1, better robustness."),
        ("Measured robustness", f"7 realistic conditions + 2 stress tests on held-out clips; drop of only {NUM['drop']} pt."),
        ("Lossless INT8", "Static per-channel quantization: 15 MB → 4.1 MB with no top-1 loss."),
        ("One pipeline, two platforms", "Kotlin port of the feature code ships with a golden-file test against the Python reference."),
        ("Fully offline and private", "Camera frames never leave the device; works in airplane mode."),
    ]
    for i, (h, d) in enumerate(diffs):
        cx, cy = 0.5 + (i % 2) * 4.6, 1.15 + (i // 2) * 1.35
        box(s, cx, cy, 4.4, 1.2)
        circle_num(s, cx + 0.18, cy + 0.2, i + 1, fill=AMBER)
        text(s, cx + 0.75, cy + 0.12, 3.5, 1.0, [[(h, 14, WHITE, True)], [(d, 11, MUTED, False)]])

    # ---------- 7. demo & setup ----------
    s = S[6]
    for sh in list(s.shapes):
        if sh.name == "Text Box 36":
            remove(sh)
    box(s, 0.5, 1.15, 5.2, 2.75, fill="070A12", line="22304F", adj=0.04)
    cmds = [("# Python 3.11", MUTED), ("pip install -r src/requirements.txt", WHITE), ("", WHITE),
            ("# one clip → JSON", MUTED), ("python src/main.py path/to/clip.mp4", WHITE), ("", WHITE),
            ("# predictions.jsonl + accuracy", MUTED),
            ("python src/scripts/evaluate.py --videos <dir>", WHITE), ("", WHITE),
            ("# robustness + latency", MUTED),
            ("python src/scripts/evaluate.py --robustness --stress", WHITE),
            ("python src/scripts/benchmark.py --thermal-minutes 10", WHITE)]
    text(s, 0.7, 1.27, 4.9, 2.6, [{"runs": [(t, 10.5, c, False)], "space_after": 0} for t, c in cmds])
    box(s, 0.5, 4.05, 5.2, 1.05)
    text(s, 0.7, 4.15, 4.9, 0.9, [[("Repository", 13, WHITE, True)],
                                  [("src/  code + INT8 model  ·  results/  predictions + results.csv  ·  "
                                    "android/  app  ·  docs/  deck + demo video", 10.5, MUTED, False)]])
    for tb in s.shapes:
        pass
    box(s, 6.0, 1.15, 3.5, 1.85)
    text(s, 6.2, 1.28, 3.1, 1.65, [[("Demo video", 14, WHITE, True)],
                                   [("Live signing on an Android phone: word, top-5, sentence, stats overlay", 11, MUTED, False)],
                                   [(REPO, 11, CYAN, False)]])
    box(s, 6.0, 3.2, 3.5, 1.9)
    text(s, 6.2, 3.33, 3.1, 1.7, [[("Android app", 14, WHITE, True)],
                                  [("Open android/ in Android Studio → Run. Models are bundled in assets/; "
                                    "no network needed.", 11, MUTED, False)],
                                  [("minSdk 26 · CameraX · MediaPipe · ONNX Runtime", 10, CYAN, False)]])
    # monospace for the command block
    for sh in s.shapes:
        if sh.has_text_frame and "pip install" in sh.text_frame.text:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name = "Courier New"

    # ---------- 8. brownie ----------
    s = S[7]
    for sh in list(s.shapes):
        if sh.name == "Text Box 36":
            remove(sh)
    set_run_text(by_name(s, "Text 2"), "BROWNIE FEATURES")
    feats = [
        ("Live Android app", "Real-time word + top-5 with confidence, hand-skeleton overlay, front/back camera"),
        ("Sentence building", "Signs join into a sentence; tap a top-5 alternative to correct a word; undo/clear"),
        ("Automatic sign segmentation", "Detects when hands rise into the signing space and drop again: no buttons needed"),
        ("On-phone metrics overlay", "p50/p95 latency, FPS, power (ML on vs off), battery temperature, thermal drift"),
        ("Stress tests", "Occlusion and very low light beyond the official conditions (drop 5 pt)"),
        ("Extra training data", "ASL Citizen integrated for words it shares with WLASL (1,385 words)"),
    ]
    for i, (h, d) in enumerate(feats):
        cx, cy = 0.5 + (i % 3) * 3.05, 1.15 + (i // 3) * 2.0
        box(s, cx, cy, 2.85, 1.85)
        circle_num(s, cx + 0.2, cy + 0.2, "★" if i < 4 else "+", fill=CYAN if i < 4 else AMBER)
        text(s, cx + 0.2, cy + 0.72, 2.5, 1.1, [[(h, 13, WHITE, True)], [(d, 10.5, MUTED, False)]])

    # ---------- 9. thank you ----------
    s = S[8]
    ty = by_name(s, "Text 9")
    ty.left, ty.width = Inches(0.5), Inches(9.0)
    ty.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    for sh in s.shapes:
        if sh.name == "Text 25" and sh.has_text_frame:
            tf = sh.text_frame
            tf.text = ""
            r = tf.paragraphs[0].add_run()
            r.text = f"gods_plan  ·  VIT Vellore  ·  {REPO}"
            r.font.size, r.font.color.rgb, r.font.name = Pt(11), rgb(WHITE), FONT
            tf.paragraphs[0].alignment = PP_ALIGN.CENTER
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    text(s, 0.5, 3.2, 9.0, 0.5, [{"runs": [("EchoGesture: sign language, understood on-device", 14, MUTED, False)],
                                  "align": PP_ALIGN.CENTER}])
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
