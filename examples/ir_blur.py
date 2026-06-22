"""Step 1 test: build the line-based IR and read off the delay-line count.

Run:  python examples/ir_blur.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from np2hw.ir import ImageStreamer, Image2D, Indexer, ImageOp


def banner(t): print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


# --- The [1,1]x[1,1] (2x2 box) blur, exactly as in idea.txt -----------------
banner("2x2 box blur  (separable: horizontal [1,1] then vertical [1,1])")
s = ImageStreamer()
img = Image2D("img", width=1920, height=1080, bits=8)
i1 = Indexer(start=0, step=1)
i2 = Indexer(start=1, step=1)
_add = ImageOp("add", i1, i2)

l1  = s.line(img, i1)           # line at vertical offset 0
l2  = s.line(img, i2)           # line at vertical offset 1
lb1 = s.horizontal(_add, l1)   # l1[x] + l1[x+1]
lb2 = s.horizontal(_add, l2)   # l2[x] + l2[x+1]
out = s.vertical(_add, [lb1, lb2])   # lb1 + lb2   <-- the corrected last step
s.report(out)
assert s.analyze(out)["line_buffers"] == 1, "2x2 box needs exactly 1 line buffer"
assert out.bits == 10, "8b -> +1 (h) -> +1 (v) = 10b for a sum of 4 pixels"
print("\nOK: 1 line buffer, 10-bit output.")


# --- A 3-tap vertical (no horizontal) -> 2 line buffers ---------------------
banner("3-tap vertical blur  (offsets 0,1,2)  -> expect 2 line buffers")
s = ImageStreamer()
img = Image2D("img", 1920, 1080, 8)
lines = [s.line(img, Indexer(start=k)) for k in range(3)]
vadd3 = ImageOp("add", Indexer(0), Indexer(1), Indexer(2))
out = s.vertical(vadd3, lines)
s.report(out)
assert s.analyze(out)["line_buffers"] == 2
print("\nOK: 2 line buffers (V-1 for a 3-row window).")


# --- Cascade of two 3-tap verticals -> line-buffer count composes -----------
banner("cascade: 3-tap vertical, then another 3-tap vertical -> expect 4")
s = ImageStreamer()
img = Image2D("img", 1920, 1080, 8)
stage1 = [s.line(img, Indexer(start=k)) for k in range(3)]
mid = s.vertical(ImageOp("add"), stage1)          # output lag 0
# read the intermediate line at 3 vertical offsets for the 2nd stage:
mid_offsets = [s.line(mid, Indexer(start=k)) for k in range(3)]
out = s.vertical(ImageOp("add"), mid_offsets)
print(f"line buffers: {s.analyze(out)['line_buffers']}  (2 + 2)")
assert s.analyze(out)["line_buffers"] == 4
print("OK: delay-line count composes across stages.")
