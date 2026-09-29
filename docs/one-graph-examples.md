# One graph: worked examples

Companion to [one-graph.md](one-graph.md). Written by hand, BEFORE any code,
to test the model: each case is a NumPy model, the graph it traces to, and
the hardware that graph lowers to. A case that reads badly shows where the
model is wrong. Each ends with what it settles and what it leaves open,
numbered by the note's Decisions.

## Notation

Variables are declared first, each with its count and whether it drives
(a counter) or is unrolled (lanes):

    vars  r : H   drives        c : W   drives

A value's type is its AXES -- the variables it spans -- then its container
and exact range:

    %3 = model.mul(%1, %2) : (r, c) u16 [0, 65025]

An access takes one index per axis of the buffer it reads; its result spans
the variables that appear. Indexes are expressions -- nodes, like any other
arithmetic -- and are written inline here for brevity:

    %5 = model.access(%2) {rows: r + 1, cols: c + 2} : (r, c) u8 [0, 255]

stands for `model.access(%2, %rows, %cols)` with `%rows` the canonical node
for `r + 1` and `%cols` for `c + 2`. After canonicalising, equal indexes are
one node, so the nine accesses of a 3x3 stencil use three row indexes and
three column indexes, each shared by three accesses.

Hardware nodes are written the same way without axes -- a hardware value is
one wire, one beat -- and state is named: `hw.reg name <- next if enable`.

---

## 1. A pointwise gain

```python
def gain(img, g):
    return ((img.astype(np.uint16) * g) >> 4).clip(0, 255).astype(np.uint8)
```

**Model level**

    vars  r : H   drives        c : W   drives
    %0 = model.source {image=img}          : (r, c) u8   [0, 255]
    %1 = model.astype(%0) {u16}            : (r, c) u16  [0, 255]
    %2 = model.param {gain, u8}            : ()     weak [0, 255]
    %3 = model.mul(%1, %2)                 : (r, c) u16  [0, 65025]
    %4 = model.shr(%3) {4}                 : (r, c) u16  [0, 4064]
    %5 = model.clip(%4) {0, 255}           : (r, c) u16  [0, 255]
    %6 = model.astype(%5) {u8}             : (r, c) u8   [0, 255]

No access at all: data math on a buffer applies at every position of its
axes. `%2` has no axes and broadcasts.

**Canonical:** both `astype`s drop -- the ranges fit their containers, so
nothing wraps. `>> 4` stays a shift: its operand is data, not a word being
split.

**Hardware**

    counters   r_row, r_col          from r, c: the only driving variables
    datapath   mul(in_data, param_gain) -> shr 4 -> clip [0, 255]
    stage      out_data <- datapath       if accepted         (no reset)
    flags      out_sof  <- first(r) & first(c)                 (reset)
               out_eol  <- last(c)
               out_last <- last(r) & last(c)

`first` and `last` are each driving variable's boundaries, from its count.

**Settles.** Scalars broadcast (3); the container decides wraps (5); the
control comes from `r` and `c` (7). **Open.** None new.

---

## 2. A 3x3 stencil with replicated edges

```python
def blur(img):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((x[:-2, :-2] + 2 * x[:-2, 1:-1] + x[:-2, 2:]
           + 2 * x[1:-1, :-2] + 4 * x[1:-1, 1:-1] + 2 * x[1:-1, 2:]
           + x[2:, :-2] + 2 * x[2:, 1:-1] + x[2:, 2:]) // 16).astype(np.uint8)
```

**Model level**

    vars  r : H   drives        c : W   drives
    %0 = model.source {image=img}                  : (H, W) u8
    %1 = model.astype(%0) {u16}                    : (H, W) u16
    %2 = model.pad(%1) {rows: 1, 1  cols: 1, 1  edge}  : (H+2, W+2) u16
    %3 = model.access(%2) {rows: r + 0, cols: c + 0}   : (r, c) u16 [0, 255]
    %4 = model.access(%2) {rows: r + 0, cols: c + 1}   : (r, c) u16 [0, 255]
      ...                                              (nine accesses)
    %20 = model.add(...)                            : (r, c) u16 [0, 4080]
    %21 = model.floordiv(%20) {16}                  : (r, c) u16 [0, 255]
    %22 = model.astype(%21) {u8}                    : (r, c) u8

The padded buffer's axes are POSITIONS; the slices are accesses mapping the
output's variables onto them. `x[:-2, 1:-1]` is `rows: r + 0, cols: c + 1`.

**Canonical:** the adds and constant multiplies become one linear sum over
the nine accesses, weights 1 2 1 / 2 4 2 / 1 2 1; `// 16` a shift. The
weights are powers of two, so every term is a shift -- wiring -- and the
cost is the adders: nine terms, EIGHT adders, over line buffers of 8-bit
pixels.

**What the written form decides.** Factoring the separable weights inside
the window saves nothing: three row sums `a + 2b + c` (two adders each)
and a column sum (two) are eight adders too. The saving needs the row sum
computed ONCE a pixel and stored, which a model can write in NumPy:

```python
x = np.pad(img.astype(np.uint16), 1, mode="edge")
h = x[:, :-2] + 2 * x[:, 1:-1] + x[:, 2:]      # horizontal: once a pixel
out = (h[:-2] + 2 * h[1:-1] + h[2:]) // 16     # vertical: on the stored sums
```

FOUR adders -- two horizontal, two vertical -- and the line buffers hold
`h`, 10-bit row sums, instead of 8-bit pixels: a quarter more storage for
half the adders. Each form gets the circuit it states, bit-exact because
it is what NumPy computes; the author chooses by writing one or the other.
Lowering does not choose for them.

**Window rule:** one group -- every access uses `r` and `c` with step 1.
Starts on `r` spread 0..2: two line buffers. Starts on `c` spread 0..2: a
shift register two deep. The pad's edge rule is replicate.

**Hardware**

    counters   r_row, r_col
    memory     linebuf: 2 x W, one write port, one registered read port
    shift      two taps per row, three rows
    edge       the pad: rows and columns past the frame read the nearest
               row or column; the output lags the input one row and one
               column, and the frame's end is FLUSHED
    datapath   the nine-term sum: eight adders -> shr 4
               (the two-pass form: two adders, line buffers of row sums,
               two adders)
    stage, flags   as in 1, from the OUTPUT position

**Settles.** Pad is assembly -- an axis extended by an edge rule -- and
slices on it are access (4); storage comes from the spread (window rule).
The output trails the input by the lag the spread gives -- one row and one
column -- and the outputs past the input's end (the last column of each
line, the last line of the frame) are produced in FLUSH cycles: blanking,
where the source has it, or a cycle holding the next frame back through
`in_ready` where it does not. The output's counters trail the input's by
that lag (7).

---

## 3. A CFA phase plane, written two ways

The same black-level stage: a per-CFA-colour offset.

```python
# (a) strided slices written into a canvas
def black_a(img, py, px, off):
    v = img.astype(np.int32)
    out = np.empty_like(v)
    for i, rows in enumerate((py, 1 - py)):
        for j, cols in enumerate((px, 1 - px)):
            out[rows::2, cols::2] = (v[rows::2, cols::2] + off[i, j]).clip(0, 4095)
    return out.astype(np.uint16)

# (b) the phase computed from the pixel's position
def black_b(img, py, px, off):
    v = img.astype(np.int32)
    y, x = coords(img)
    i, j = (y + py) & 1, (x + px) & 1
    o = np.where(i == 0, np.where(j == 0, off[0, 0], off[0, 1]),
                         np.where(j == 0, off[1, 0], off[1, 1]))
    return (v + o).clip(0, 4095).astype(np.uint16)
```

**(a), model level**

    vars  r : H  drives   c : W  drives   q : H/2   p : W/2
    %1 = model.astype(%0) {i32}                          : (r, c)
    %2 = model.access(%1) {rows: 2q + py, cols: 2p + px}  : (q, p)
    %3 = model.add(%2, off00) ; %4 = clip(%3)            : (q, p)
      ... three more planes, starts 1 - py and 1 - px ...
    %9 = model.canvas(%1, %4, %5, %6, %7)
           {planes: rows 2q + py, cols 2p + px
                    rows 2q + py, cols 2p + 1 - px
                    rows 2q + 1 - py, cols 2p + px
                    rows 2q + 1 - py, cols 2p + 1 - px}  : (r, c)

**(b), model level**

    vars  r : H  drives   c : W  drives
    %2 = model.coords(%0)                  : (r,) and (c,)   -- the variables
    %3 = model.and(model.add(%2#0, py), 1) : (r,)  -- (r + py) % 2
    %4 = model.and(model.add(%2#1, px), 1) : (c,)  -- (c + px) % 2
    %5 = model.where(...)                  : (r, c) -- one of four registers
    %6 = model.add(%1, %5) ; clip          : (r, c)

**Canonical -- both must become this:**

    %s = select by (r + py) % 2, (c + px) % 2 :
           [off00, off01, off10, off11]      : (r, c)  weak  -- a register
    %6 = model.add(%1, %s) ; clip [0, 4095]  : (r, c)

In (a) the canvas is a select by the remainders of the planes' starts: four
planes that tile the lattice, each written from the SAME plane it reads.
The planes never coincide -- each pixel is in exactly one -- and compute the
same operations with different registers, so the share pass merges their
four datapaths into one with a selected operand: four datapaths never in
use together are waste, and each pixel still gets its plane's arithmetic. In (b) `& 1` of an
index is its remainder -- index math, not data -- and the nested `where` on
remainders is the same select.

**Hardware**

    datapath   add(in, sel(param_off00..11 by row[0]^py, col[0]^px)) -> clip
    no storage -- every access is at step 2 with the starts tiling the
               lattice, so the "planes" are one datapath and a select

**Settles.** A canvas whose planes tile the lattice, and a nested `where`
on remainders, both canonicalise to one SELECT BY REMAINDERS (4); `& 1` and
`% 2` on a variable split it (1); a register in a start is a run-time
remainder (1).

---

## 4. A two-channel word, mixed and repacked

```python
def mix(img, m):                    # 16-bit word: two 8-bit channels
    v = img.astype(np.int64)
    a = v & 0xFF
    b = (v >> 8) & 0xFF
    aa = ((a * m[0, 0] + b * m[0, 1]) >> 8).clip(0, 255)
    bb = ((a * m[1, 0] + b * m[1, 1]) >> 8).clip(0, 255)
    return (aa + bb * 256).astype(np.uint16)
```

**Model level (as written)**

    %1 = model.astype(%0) {i64}   : (r, c) i64 [0, 65535]
    %2 = model.and(%1, 255)       : (r, c)     [0, 255]
    %3 = model.shr(%1, 8) ; %4 = model.and(%3, 255)
      ... data math ...
    %9 = model.add(%aa, model.mul(%bb, 256))  : (r, c) [0, 65535]

**Canonical:** `& 0xFF` of a value known in `[0, 65535]` is an access on its
bit axis, `bits[0:8]`; `(v >> 8) & 0xFF` is `bits[8:16]`. `aa + bb * 256`,
with `aa` in `[0, 255]`, is assembly: `bits[0:8] = aa, bits[8:16] = bb`.

**Hardware**

    wiring     in_data[7:0], in_data[15:8]         -- no AND gates
    datapath   two 2x2 dot products, shift, clip
    wiring     out = {bb, aa}                      -- no adder

**Settles.** Bit access and bit assembly turn masks, shifts and packing
into wiring; `& k` is a bit access only when `k` is a low-bit mask and the
operand is known non-negative, decided in canonicalising, while the model
graph records what was written (2, 4).

---

## 5. Two pixels a clock, by reshape

```python
def pair_sum(img):                  # horizontal pair sum: halves the width
    p = img.astype(np.uint16).reshape(H, W // 2, 2)
    return p[:, :, 0] + p[:, :, 1]
```

**Model level**

    vars  r : H  drives   q : W/2   k : 2
    %1 = model.astype(%0) {u16}                       : (r, c)
    %2 = model.access(%1) {rows: r, cols: 2q + k}     : (r, q, k)  -- the reshape
    %3 = model.access(%2) {k: 0}                      : (r, q)
    %4 = model.access(%2) {k: 1}                      : (r, q)
    %5 = model.add(%3, %4)                            : (r, q)

**Two hardwares, and the choice is not the model's.** Whether `k` drives or
is unrolled depends on how the SOURCE arrives:

- **one pixel a beat** -- the stream's contract today. Then `c` drives, and
  with it `k` (the faster part of `c`): the access `k: 0` against `k: 1` is
  a spread of one on a driving variable -- a one-pixel shift register --
  and the output is valid every second beat (the width halves).
- **two pixels a beat** -- a source declared `(H, W/2, 2)` with its last
  axis in the word. Then `k` is unrolled: the two accesses are the two
  halves of the word, and the adder runs once a beat with no storage.

**Settles.** Reshape is access with a split index, and moves nothing; drive
or unroll is decided at the stream boundary, by the source's declared
packing (1).

---

## 6. Depth from the last four frames

```python
def depth(video, ph):               # video: (T, H, W); one phase a frame
    v = video.astype(np.int32)
    s = [v[k + 3 - ph] for k in range(4)]   # the four frames, by phase
    i, q = s[0] - s[2], s[1] - s[3]
    return atan2_q(i, q)                   # CORDIC, as in tofquad
```

(Illustrative: the history is the last four frames, and which one carries
which reference phase follows the register `ph`.)

**Model level**

    vars  t : T  drives (outermost)   r : H  drives   c : W  drives
    %1 = model.astype(%0) {i32}                  : (t, r, c)
    %2 = model.access(%1) {frames: t + 3 - ph}   : (t, r, c)
    %3 = model.access(%1) {frames: t + 2 - ph}   ...  (four accesses)
    %6 = model.sub(%2, %4) ; %7 = model.sub(%3, %5)
    %8.. CORDIC data math                        : (t, r, c)

**Window rule on `t`:** one group -- `t`, `r`, `c` at step 1. Starts on `t`
spread over 3: THREE FRAME STORES, derived from the reads, not declared.
The register `ph` in the starts selects which stored frame feeds which term:
a remainder chosen at run time, as the CFA phase is on rows and columns.

**Hardware**

    counters   r_frame, r_row, r_col      -- t, the outermost driver
    memory     frames: 3 x H x W, lowered to block RAM, a macro, or
               external memory by its properties
    select     which frame is which phase, by ph
    datapath   two differences -> CORDIC

**Settles.** Frame history is access on `t`, its depth from the reads; the
source is `(T, H, W)`, `t` advancing on each start of frame; the first
frames follow what the model wrote -- a slice that starts at `3` gives no
output until three frames are stored, a pad on `t` defines them (6).

---

## What the examples decided

Scalars broadcast by variable (3); a container decides where a wrap is
(5); the control comes from the driving variables, the output's counters
trailing the input's by the derived lag and finishing in flush cycles (7);
pad is assembly and slices on it are access (4); a canvas whose planes tile
the lattice, and a nested `where` on remainders, are one select by
remainders (4); masks, shifts and packing on bits are wiring (2); reshape
is access with a split index, and drive or unroll is the source's packing
(1); frame history is access on `t`, its depth derived and its first
frames as the model wrote them (6). All seven decisions in the note are
settled.
