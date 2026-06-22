"""Live viewer for np2hw: capture frames -> run an ISP model -> show input|output.

Capture sources (chosen by the CLI, all yield 8-bit grayscale HxW ndarrays):
  camera  -- webcam via OpenCV (needs the [camera] extra: opencv-python)
  screen  -- screen grab via Pillow (no extra dep)
  test    -- a synthetic moving gradient (for headless verification)

The model in the middle is pure NumPy (same function that compiles to RTL); the
capture/display are runtime harness only -- not NumPy, not hardware.

Output: a live Tkinter window (input | output side by side), or, with snapshot=,
a single montage image written to a file (headless -- no GUI/camera needed).
"""
import numpy as np


# --------------------------------------------------------------------------- #
# Capture sources -- each is an iterator of (H, W) uint8 grayscale frames
# --------------------------------------------------------------------------- #

def _to_gray(arr, size):
    from PIL import Image
    im = Image.fromarray(arr).convert("L")
    if size:
        im = im.resize(size)                              # (W, H)
    return np.asarray(im, dtype=np.uint8)


def source_camera(size, index=0):
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit("could not open camera (try a different --camera index)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield _to_gray(frame[:, :, ::-1], size)       # BGR->RGB->L
    finally:
        cap.release()


def source_screen(size, region=None):
    from PIL import ImageGrab
    while True:
        im = ImageGrab.grab(bbox=region).convert("L")
        if size:
            im = im.resize(size)
        yield np.asarray(im, dtype=np.uint8)


def source_test(size, frames=None):
    W, H = size or (160, 120)
    base = (np.add.outer(np.linspace(0, 200, H), np.linspace(0, 200, W)) % 256
            ).astype(np.uint8)
    t = 0
    while frames is None or t < frames:
        yield np.roll(base, t * 3, axis=1)
        t += 1


def open_source(name, size, camera=0, region=None, frames=None):
    if name == "camera":
        return source_camera(size, camera)
    if name == "screen":
        return source_screen(size, region)
    if name == "test":
        return source_test(size, frames)
    raise SystemExit(f"unknown --source {name!r} (camera|screen|test)")


# --------------------------------------------------------------------------- #
# Compose input|output side by side (uint8 grayscale)
# --------------------------------------------------------------------------- #

def montage(inp, outp):
    """Stack input and output side by side; pad heights, 2px separator."""
    h = max(inp.shape[0], outp.shape[0])
    def pad(a):
        out = np.zeros((h, a.shape[1]), np.uint8)
        out[:a.shape[0], :a.shape[1]] = a
        return out
    sep = np.full((h, 2), 128, np.uint8)
    return np.concatenate([pad(inp), sep, pad(outp)], axis=1)


# --------------------------------------------------------------------------- #
# Run loops
# --------------------------------------------------------------------------- #

def snapshot(src, run, path, n=8, pv=None):
    """Headless: grab n frames, process, save a vertically-stacked montage."""
    from PIL import Image
    rows = []
    for i, frame in enumerate(src):
        rows.append(montage(frame, run(frame, pv)))
        if i + 1 >= n:
            break
    w = max(r.shape[1] for r in rows)
    grid = np.concatenate(
        [np.pad(r, ((0, 0), (0, w - r.shape[1]))) for r in rows], axis=0)
    Image.fromarray(grid, "L").save(path)
    return grid.shape


def _param_range(p):
    if getattr(p, "signed", False):
        return -(1 << (p.bits - 1)), (1 << (p.bits - 1)) - 1
    return 0, (1 << p.bits) - 1


def _fps(times):
    """Frames/sec from a window of recent timestamps (>=2)."""
    return (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 else 0.0


def live(src, run, params=None, init_pv=None, title="np2hw view", scale=3):
    """Live Tkinter window: input | output, with a slider per Param for real-time
    control, and end-to-end FPS in the title bar. Closes on window close."""
    import time
    from collections import deque
    import tkinter as tk
    from PIL import Image, ImageTk
    params = params or []
    stamps = deque(maxlen=30)                            # recent frame times -> FPS
    init_pv = dict(init_pv or {})
    win = tk.Tk(); win.title(title)
    label = tk.Label(win); label.pack()
    vars_ = {}
    for p in params:                                     # one control per register
        if p.bits == 1:                                  # 1-bit -> checkbox (bool)
            var = tk.IntVar(value=int(init_pv.get(p.name, 0)))
            tk.Checkbutton(win, text=p.name, variable=var,
                           onvalue=1, offvalue=0).pack(anchor="w")
        else:                                            # multi-bit -> slider
            lo, hi = _param_range(p)
            var = tk.IntVar(value=int(init_pv.get(p.name, (lo + hi) // 2)))
            tk.Scale(win, from_=lo, to=hi, orient=tk.HORIZONTAL, length=400,
                     label=f"{p.name} [{lo}..{hi}]", variable=var).pack(fill="x")
        vars_[p.name] = var
    state = {"run": True}
    win.protocol("WM_DELETE_WINDOW", lambda: state.update(run=False))

    def tick():
        if not state["run"]:
            win.destroy(); return
        try:
            frame = next(src)
        except StopIteration:
            win.destroy(); return
        pv = {n: v.get() for n, v in vars_.items()}      # current slider values
        m = montage(frame, run(frame, pv))
        im = Image.fromarray(m, "L")
        im = im.resize((im.width * scale, im.height * scale))
        photo = ImageTk.PhotoImage(im)
        label.configure(image=photo); label.image = photo
        stamps.append(time.perf_counter())               # end-to-end loop rate
        win.title(f"{title} — {_fps(stamps):.1f} fps")
        win.after(1, tick)

    win.after(1, tick); win.mainloop()
