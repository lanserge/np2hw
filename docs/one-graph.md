# One graph: indexes, access, data

Status: in progress, its decisions settled (see the end). The graph core,
the tracer, canonicalising, the pointwise family's lowering, windows,
stages cut to a clock, and reads across a video's frames are built in
`np2hw.core`, and `to_ir` hands every
model to it first; a model it does not reach yet is lowered by np2hw's
present compiler, the reference it is measured against -- except a video,
which the reference has no frame axis for. `NP2HW_CORE=off` forces the reference, `NP2HW_CORE=strict`
refuses the fallback, and `NP2HW_CORE_LOG` names a file recording which
path each model took.

np2hw compiles in two stages. **Tracing** records a NumPy model as ONE graph
-- what the model computed, nothing decided. **Lowering** rewrites that graph,
pass by pass, into a hardware graph, which is printed as Verilog. This note
is the model the graph expresses, the shape of the graph, what lowering does,
and how the new compiler is built and judged.

## Why

np2hw's present compiler traces a model into one of six representations --
a linear chain, an expression DAG, a phase canvas, a channel stack, a mux,
a line IR -- chosen at the operator that produced each value, and
`generate()` dispatches on which. Each works; the defects live in the SEAMS
between them. Every recent one was a value that had taken the wrong form for
what came next: a constant bias the chain could not hold; a shift refused
where the equal division worked, because a register multiply had been
committed early to a tap coefficient; a table index accepted by one form
converter and refused by another; window expressions possible only under a
phase select; register taps and edge handling meeting in one emitter only.

Moving the compiler onto a graph in place was tried, and kept bending the
graph around the old forms. So the new compiler, `np2hw.core`, is written
from this model; the present one, whose tracer already records a model as
a graph (`np2hw.trace`) and lowers it through the older forms, serves as
its reference until nothing needs it.

## The model: three layers

A model is written in three layers, and the graph holds all three.

### 1. Index math

An index starts from a count:

    i = Index(8)          # a variable: 0, 1, ..., 7

and arithmetic builds on it:

    2 * i                 # step 2:          0, 2, ..., 14
    2 * i + 1             # step 2, start 1: 1, 3, ..., 15
    3 * (2 * i + 1)       # 6 * i + 3 -- a multiply scales the start too

Each `Index(count)` creates a VARIABLE with its own identity: `i` is not `j`,
even with the same count. Index math builds ordinary graph nodes on
variables; its affine form -- start and steps -- is derived, not stored. Adding two indexes is index math like any other:
`2 * i + j`, with `i = Index(W // 2)` and `j = Index(2)`, is an index over
both variables. A start may also hold REGISTERS, with integer coefficients --
`2 * Index(n) + p.py` reads the rows the register selects, `1 - p.py` the
other half -- and the index stays affine: a register never multiplies a
variable or another register.

Floor division and remainder by a constant are index math too: `col // 2` and
`col % 2` are the `q` and `r` of `col = 2 * q + r`.

### 2. Access

A buffer indexed with indexes gives a buffer, axis after axis:

    buf[row_index, col_index]

and the result has ONE AXIS PER VARIABLE in the indexes. An index is an
expression -- variables and the arithmetic on them -- and what it is decides
what the access is: affine in the variables, a streamed read; computed from
data, a lookup, as `knots[seg]` reads a table. That is also where reshape
comes from, without being an operation of its own:

- `line[2 * i + j]`, `i` in `0 .. W/2 - 1`, `j` in `0 .. 1`, is a `(W/2, 2)`
  buffer of pixel pairs -- a split;
- `flat[W * i + j]` is an `(H, W)` view of a one-dimensional stream -- a
  merge.

A NumPy slice is access with an index written as a slice: `x[1:-1]` is
`x[Index(n - 2) + 1]`, `x[py::2]` is `x[2 * Index(n // 2) + py]`. A NumPy
`reshape` of a C-ordered array is access with a split or merged index, and
moves no element. `coords(img)` hands the model the variables of `img`'s own
axes as values, and `make_coords` a declared output raster's; affine
arithmetic on them stays index math, so `img[y2 + 2, x2 + 2]` is access, a
crop.

A value is a BUFFER with named axes, and its shape is derived: a source's
shape is known and every operation's result follows from its operands. The
streamed source is the whole buffer:

    frames x rows x cols x channels, each element a word of bits

The bits of an element are accessed too: `x >> 8` and `x & 0xFF`, with
constants that make them so, are `bits[8:]` and `bits[0:8]` of the value's
representation, and channels packed in a word are its channel axis laid out
along the bits. The inverse of access ASSEMBLES: `np.stack` adds an axis, a
canvas writes planes back into a frame, repacking concatenates bits, and
`np.pad` extends an axis by an edge rule. Registers and constants are
buffers with no axes, broadcast where they meet one.

### 3. Data math

Arithmetic on what was accessed: `+ - *`, `//` and `>>` by constants,
`clip`, `abs`, comparisons, `where`. Every
value carries two facts: its exact integer RANGE, from which width and sign
follow, and its NumPy CONTAINER -- the dtype NumPy holds it in under
NumPy 2's rules (a Python int or a register is weak and adopts the other
operand's dtype). The container says where NumPy wraps: `uint8 - 128`
wraps, and the hardware wraps exactly there and nowhere else.

A variable used as data -- `row * row`, a lookup by position -- is the
counter's value on a wire, with the range its count gives.

### In hardware, each layer has one meaning

| layer | becomes |
| --- | --- |
| a variable that DRIVES | a counter |
| a variable that is UNROLLED | parallel lanes |
| index math | wiring on the counters: `2 * q + r` is `q`'s bits then `r` |
| access on a driving variable | storage, sized by the spread of its starts |
| access on an unrolled variable | wiring: a bit or lane select |
| data math | the datapath |

Three variables drive, iterated in time: columns (one step a beat), rows
(one a line) and, outside them, frames (one a frame -- where registers
commit, and what a frame history reads across). Every other variable is
unrolled: channels, bits, the taps of a window (each a constant offset, so
a window is parallel reads), the slots of a frame history, the leaves of a
register matrix. What a spread of starts costs follows from where the axis
sits in the streaming order:

| axis | arrives | a spread of starts costs |
| --- | --- | --- |
| frames | one frame after another | frame stores |
| rows | one line after another | line buffers |
| columns | one pixel after another | a shift register |
| channels, bits | all in the same beat | nothing: a select |

A reshape keeps the element order, which IS the arrival order, so on a
stream it is free: columns split into `(W / 2, 2)` make the pair `col >> 1`
and the lane `col & 1`. A transpose changes the order, and the same rule
prices it: its spread of starts spans whole frames.

A stream's control is therefore not a block written into every module: a
module has a row counter and a column counter because its graph indexes
rows and columns, and a frame counter when it indexes frames. The framing
flags are the counters' boundaries -- first and last along each driving
variable, from its count.

## The graph

### A node

Every node has the same six fields, whatever it is:

    kind      namespaced: "model.add", "hw.reg"
    operands  values: (node, result index)
    attrs     plain data: ints, strings, tuples -- a shift amount, a name
    results   one type per result: a range and a container
    regions   subgraphs -- empty today; reserved for hierarchy
    loc       the model's source line

A walk, a hash, a printer and a verifier work on any kind, including one
added later. A value is (node, result index), so a node with several
results -- `coords`' row and column, a CORDIC's magnitude and angle -- needs
no redesign.

**Kinds are registered, not listed.** A kind declares its TRAITS (`pure`,
`stateful`, `commutative`, `constant`, `reads_register`) and its HOOKS
(range inference, folding, verification). **Passes select by trait, never by
kind name**: adding a kind is one declaration in one place.

**Facts in the graph, choices in lowering.** A node carries facts: its
range, its container, its indexes, its line. It never carries a choice --
"this register is a tap coefficient", "use that emitter". A choice recorded
in the graph is the six forms again.

**Immutable, and hash-consed where it shares.** A pass takes a graph and
returns one; it never edits a node. The model graph as traced is a faithful
record and merges nothing; from canonicalising on, identical nodes are made
identical at construction. A location is provenance, not identity: when
nodes from two lines merge, the node keeps both. Recording the model's line
costs about 1.5 us a node, measured -- compile time spent for messages and
Verilog comments that name the line.

**Storage is named; reads are values.** A register, a table, a memory is ONE
named declaration; each use is a read naming it. This is also how a stream's
cycle through its registers -- the output-valid register enables itself
through the stall -- is said in a graph that has no cycles.

**A level is a legality rule.** Model kinds are `model.`, hardware kinds
`hw.`; the verifier checks after each pass, and lowering ends with nothing
left that is not `hw.`.

**Everything prints, deterministically**, values numbered in walk order so
two dumps diff cleanly, every node with its kind, operands, attributes and
facts -- register declarations included:

    %3 = model.mul(%1, %2) {dtype=('u', 16)} : [0, 65025] loc("gain.py":7)

### The model level

What tracing builds: a source; variables; index expressions, which are
ordinary arithmetic on variables; `access` nodes taking a buffer and one
index per axis -- a streamed read or a lookup, as the index decides;
assembly (stack, canvas, bit assembly, pad); the data operations;
registers and constants. Each node records its container. Nothing about clocks,
counters or registers exists at this level.

### The hardware level

What lowering builds, CIRCT's shape cut down:

- **pure nodes** on exact ranges -- arithmetic, comparisons, selects, the
  wrap a container imposes -- with widths explicit and no Verilog in them;
- **stateful nodes**, explicit: registers (with a reset value, or none), the
  counters of the driving variables, memories;
- **memory is one kind** with declared properties -- depth, width, ports,
  read latency, what a read returns when the same address is written in
  the same cycle. A line buffer is a memory and a counter. WHERE it lives --
  block RAM, a macro behind the memory seam, external memory -- is decided
  late, from those properties; the same choice serves a frame history;
- **ports** with np2hw's stream contract: valid/ready and `sof`/`eol`/`last`
  per stream, one token per stream, the datapath inside scheduled
  statically, everything stalling together on `ready`.

A register needs a reset only if its value is read without something that
IS reset saying it is valid: the output data register does not, since its
contents mean something only while `out_valid` says so. Measured, a reset
it did not need cost 8 LUTs a module.

## Lowering

**NumPy is the hardware language.** Lowering follows the model operation by
operation: each operation wraps at its own container, `//` and `>>` floor,
`clip` saturates where it was written, promotions happen between the steps
where NumPy makes them. The circuit is therefore bit-exact by
CONSTRUCTION, not by checking, and it is the circuit the model states: one
that recomputes at every tap gets that; one that computes once and indexes
the result gets storage of the result. The author optimises by writing the
NumPy -- and gets what they wrote.

Lowering does not restructure. Reassociating a sum, moving an access across
a computation, factoring shared terms, hoisting register arithmetic: each
is exact only if every intermediate value keeps its range and container,
and proving that case by case is exactly the watch over bits the model
exists to make unnecessary -- a slip there is hardware that differs from
NumPy on rare inputs no example hits. So no pass changes the model's
structure. One would be admissible only with a proof of exactness for the
case at hand, formal equivalence checking for instance; that is a question
for later, not part of this design.

What lowering does do is exact node by node:

- **sharing identical nodes** -- the same operation on the same operands is
  the same value;
- **folding constants**, evaluated with NumPy's semantics, wraps included;
- **canonical index forms** -- index arithmetic is exact integer arithmetic
  on counters, and the canonical form is for RECOGNISING (the window rule,
  sharing), not for rewriting the circuit;
- **wraps exactly where the container says** -- which is following NumPy;
- **bit access for a low-bit mask or a shift of a value known
  non-negative** -- an exact equivalence;
- **planes that never coincide sharing one datapath** -- only when their
  computations are the same operations differing in constants or
  registers, so each pixel still gets exactly its plane's arithmetic;
- **dropping resets nothing needs** -- values untouched.

The passes, each visiting every shared node once:

1. **canonicalise** -- one form for recognising each thing: index forms,
   linear sums recognised, constants folded, bit access where it is exact;
   containers resolved into the wraps they impose;
2. **share** -- identical nodes become one node; planes that never
   coincide share one datapath;
3. **window** -- access on driving variables becomes storage and counters,
   by the window rule (below);
4. **rewrite** -- every remaining model node by its rule: a sum an adder
   tree in the written association, a select over a variable's remainder
   the phase mux, a lookup a mux or a constant table;
5. **schedule** -- every node assigned a stage from the device's cost table
   (kept in lowering, not on the nodes), and the stage map verified;
6. **materialise** -- every edge crossing a stage becomes a register,
   enabled by `ready`;
7. **print** -- one printing rule per hardware kind.

Built so far: canonicalise (which shares by interning, so the second pass
happens inside the first), then rewrite, control (the driving variables'
counters and flags), schedule, materialise and print. The window rule is
applied within rewrite and control: rewrite finds each window source and
its taps, control builds their line buffers, shift registers and edge rule
on a step raster the lag and the flush come from.

### The window rule

Every access names its variables, so the structure of the hardware is read
off them:

- **group** accesses by variable and step: accesses with the same driving
  variables and steps run on one lattice at one rate -- one pipeline;
- **size storage** from the spread of starts on each driving variable, per
  source, so accesses to one source share its line buffers;
- **select planes** by the start's remainder modulo the step: accesses
  whose steps and starts tile the lattice -- four step-2 reads with starts
  0 and 1 per axis are the CFA planes -- share one window, and the
  remainder picks the plane for each pixel produced. A register in the
  start is a remainder chosen at run time and changes nothing structural;
- **an extent** -- a count smaller than the source's -- becomes gating at
  the edges; a coarser step alone is the same clock with a valid pattern.
  The source's rows and columns past the last start are still taken from
  the stream, and never read;
- **compose** by substitution: a computed value read at a shifted index
  is that value computed at the index, its variables replaced, so a filter
  applied to a filter's result is one window over the source -- its spread
  the sum of the two. The inner arithmetic is computed at each tap of the
  outer one; identical taps are one node by interning, and a shifted copy
  is arithmetic again rather than a register, which a later pass may trade;
- **the edge** is the pad's: a row or column off the frame reads the
  nearest edge under `mode="edge"` and zero under a zero pad, each a select
  on the step's position, the rows past the last fed back by the flush.

On the frame axis the same rule is the frame history: the spread of starts
in `t` is the number of frame stores -- derived from the reads, never
declared -- and the latest start is the live frame. A sensor's capture
loop (a time-of-flight phase) is not a start on `t` but a scalar input,
the routing among the stored frames arithmetic on it (decision 6).

Where the rule stops: combining two lattices needs one brought onto the
other's, and upsampling would be a step below 1, which an index refuses; a
model that needs it is refused with its reason until an explicit repeat
exists.

### Reuse

Sharing that is exact comes free or from the share pass:

1. **Identical nodes.** Hash-consing makes the same operation on the same
   operands one node -- and, with indexes as expressions, the same index one
   node, so every access sharing an index is found by following that node's
   users.
2. **Planes that never coincide.** CFA planes -- each pixel in exactly one --
   computed by the same operations with different constants or registers
   share one datapath with a selected operand: several datapaths never in
   use together are waste.

Reuse that changes structure -- a partial sum shared between sums, a value
computed once and stored instead of recomputed at every tap -- is the
author's to write. The model that writes a separable filter as two passes
gets the circuit with half the adders and wider line buffers; the one that
writes the nine-term sum gets eight adders over narrow ones (worked example
2). Each is bit-exact because each is what NumPy computes.

## Building it

**From this model, not from the present compiler.** What carries over: the
examples, which are the specification -- NumPy in, bit-exact results
expected; the stream contract, which revela and every testbench are built
against; the device cost model, which is hardware knowledge; the gate.
Everything else -- the graph core, the tracer, lowering, the printer -- is
written new. The present compiler stays as the reference, and is deleted
when nothing needs it.

**Examples first.** Before code, the key cases are written out by hand as
graphs at both levels -- a pointwise gain, a 3x3 stencil, a CFA phase plane
written as a strided slice AND as `(row + py) & 1` from `coords` (the two
must come out the same graph), a channel repack, two pixels a clock by
reshape, a frame read across time. A case that reads badly shows where the
model is wrong. Then the code.

**Bit-exact by construction; the examples check the construction.** Every
rule is exact by design, so a compiled model is bit-exact without being
checked. What the examples verify is the COMPILER: that its code
implements those rules -- every defect found while building it was code
that broke a sound rule (axes read before unification, a canonical index
that changed an access's shape, operands paired across planes in the
wrong order), which no argument about the design would have caught. So a
model moves to the new compiler when its example passes, its generated
Verilog simulated against NumPy (`scripts/snapshot.py`). The check moves
as early as it can: each graph level is evaluated against NumPy before any
Verilog exists, and simulation is left for the printer, the one step a
Python check cannot reach -- formal equivalence of the printed Verilog
with the hardware graph would, later, prove it for every input. Synthesis figures of a module
that changed can be REPORTED (`--synth`), never gated: at the grain of one
small module the tools' own noise exceeds the differences -- the same
logic written two ways measured 1,395 to 1,909 ps in Yosys, and Yosys and
the open flow (openXC7) both ranked equivalent designs unlike Vivado. The
hardware truth is the real build at the board's clock, where a whole
design's timing is decided, as it always was. A port width that changes is
called out. Compile time is reported, not gated.

**Families one at a time**: pointwise, stencils, phase planes, channel
words, frame reads. Each is complete when its examples are on the new
compiler; the present one goes when all are.

## Prior art

MLIR and CIRCT settle most of the graph's shape: uniform nodes, registered
kinds with traits, passes selecting by trait, locations that survive
merges, verification between passes, levels as legality, storage named and
reads as values, memory as one kind lowered late, schedule separate from
materialisation, stall-based elasticity, a printer that holds no meaning.

Not taken, at this scale: control-flow regions, code generation for kind
definitions, nested pass managers and a pipeline language, explicit analysis
invalidation (immutable graphs memoise on identity), transactional rollback,
handshake on every operation, width inference (exact ranges are stronger).

Where np2hw goes past them: frames, lines and their flags; exact ranges and
NumPy containers; registers that commit at a frame boundary; indexes and
windows -- the model level has no counterpart in CIRCT, and the hardware
level is where the two meet, which is also what would let a second printer
emit CIRCT's `hw`/`comb`/`seq` and open formal equivalence checking.

References: <https://mlir.llvm.org/docs/> (language reference, traits,
interfaces, canonicalisation, pattern rewriting, pass management, dialect
conversion, symbols, diagnostics); <https://circt.llvm.org/docs/> (HW, Comb,
Seq, SV, Pipeline, ESI, Calyx, FIRRTL and OM dialects and rationales, the
scheduling infrastructure, Verilog generation, `circt-lec`).

## Decisions

Settled by the worked examples ([one-graph-examples.md](one-graph-examples.md))
and by the model.

### 1. Indexes are expressions

A VARIABLE is a node, made by `Index(count)`, by an axis of a source, or
by `coords`. Index math is ordinary graph arithmetic on variables -- `add`,
`mul`, `//`, `%` -- and an access takes each axis's index as an OPERAND:

    %v = model.var {count=W}
    %e = model.add(model.mul(%v, 2), %py)      # 2 * v + py
    %a = model.access(%buf, %rows, %e)

There is no separate index structure with an algebra of its own: one kind
of math, on one kind of node. Start and step are not stored; they are what
canonicalising derives.

- **The affine form is canonical.** Canonicalising rewrites an index that
  is affine in the variables to one normal form -- `sum(step_i * v_i) +
  start`, the start a constant plus register terms -- and hash-conses it.
  So the same index IS the same node, and every access through it is one
  of that node's users: the window rule groups accesses by following users,
  not by comparing indexes. Users are an analysis over the immutable graph,
  computed once per graph.
- **Splits are recognised, not stored.** `v // k` and `v % k` on a variable
  canonicalise to the two parts of `v = k * q + r`, new variables `q` (count
  `n / k`) and `r` (count `k`), every index over `v` rewritten in them. A
  count `k` does not divide is refused -- a ragged split has no lattice.
- **Register starts** are register terms of the start; one register may
  appear with coefficient +1 or -1 (the phase and its complement).
- **What the index is decides what the access is.** An index affine in the
  variables is a streamed read. An index computed from DATA is a lookup: on
  an unrolled or small axis a table -- a mux of registers, or a constant
  table -- and on a driving axis a random-access read, refused with that
  reason until such a memory exists. There is no separate node for a table
  read.
- **An access's result** spans the variables its indexes use, in the order
  their axes appear in the buffer read -- row-major order is kept.
- **Whether a variable drives or is unrolled is decided where its axis
  meets the stream boundary**, by the source's declared packing -- pixels a
  beat, channels a word. A variable split from a driving one drives in its
  outer part; its inner part drives at the source's pixel rate, or is
  unrolled when the source packs that many pixels into one beat.

### 2. Range, and access to bits

A value's type is its axes, its container and its exact range. Bits are
not an axis of the shape: a BIT ACCESS reads a field of the value's
representation in its container -- two's complement, as NumPy's `&`, `|`
and `>>` define it -- and a BIT ASSEMBLY writes fields into a word. Their
ranges are derived like any other. The tracer records `&` and `>>` as
written; canonicalising turns `& (2**k - 1)` and `>> k` into bit access
only where the operand's range is known non-negative, where the two
readings agree. A packed source declares its channels' field width and
order, and a channel access lowers to a bit access.

### 3. Scalars and broadcasting

Registers and constants are buffers with no axes. Everything else
broadcasts as NumPy does, POSITIONALLY: a value's axes are one per NumPy
dimension -- a variable, or none for a dimension of size 1 -- aligned from
the right; a size-1 dimension takes the other's; and two variables paired
on one position with equal counts are UNIFIED into one. That is how
`x[:-2] + x[1:-1]` comes to range over a single row variable, with indexes
`r` and `r + 1`, and how `coords`' `(H, 1)` and `(1, W)` make `(H, W)`.
Unequal counts on one position are refused, as NumPy refuses them.

An index over two distinct variables -- `2q + r`, the outer combination --
places each variable on its own axis first, as NumPy writes
`2 * q[:, None] + r[None, :]`: an `expand` node, a reshape inserting size-1
dimensions, which moves nothing and which the affine analysis looks
through.

### 4. Assembly

Four operations, and only these:

- **stack** -- values along a new UNROLLED variable, its count the number
  stacked;
- **canvas** -- planes written into a template through their indexes. The
  planes must cover each position exactly once (with `zeros_like`, a
  position no plane covers is 0), and each is cast into the template's
  container, as NumPy's assignment casts. Planes on step-`k` lattices whose
  starts tile the frame canonicalise to a SELECT BY REMAINDERS: one of `n`
  values chosen by variables' remainders -- the same node a nested `where`
  on `v % k` canonicalises to;
- **bit assembly** -- fields into a word; `a + b * 2**k`, with `a` in
  `[0, 2**k)`, canonicalises to it;
- **pad** -- an axis extended by an edge rule: replicate (`edge`) or zero
  (`constant`); others later.

Assignment to arbitrary regions is out of scope.

### 5. Containers

Every model node carries its container, as traced: NumPy 2's rules, a
Python int or a register weak. Canonicalising inserts a WRAP exactly where
a value's range leaves its container, and after it containers play no part:
hardware widths come from ranges alone. `astype` is a change of container,
a wrap when the range leaves it.

### 6. The frame axis

A video source is `(T, H, W)`, `t` its outermost driving variable; frames
arrive as the stream's frames, `t` advancing on each start of frame.
Access across time is access on `t`, and everything spatial carries over:

- the history's depth is the spread of starts on `t` -- frame stores,
  derived, never declared;
- a sensor's capture loop is a SCALAR INPUT -- the live frame's place in
  the loop, constant for the frame -- and which stored frame carries which
  phase is arithmetic on it, a lookup and a select. Where the input comes
  from -- the stream's frame phase, read from each frame's header; a
  register; a counter -- is bound where the design is composed (a node's
  input bindings in the pipeline's JSON), never said in the model, which
  is math on its inputs and nothing else;
- the start of the history follows what the model wrote: a slice that
  leaves the first frames out (`v[3:]`) makes fewer output frames, and the
  first outputs come once the history is full; a pad on `t` defines them by
  its edge rule, from the first frame.

**Where the frames are stored.** The reads on `t` state the store's
requirement completely: its depth (the spread of starts), its size (the
frame's positions, the sample's width) and its access -- every accepted
pixel writes its sample and reads the same position, and any spatial
spread around it, in each past frame. All of it in raster order, so every
address the store will need is known ahead: it is a counter.

A frame store is storage like that on every driving axis -- the reuse of
what was read -- one level further out: a shift register holds the last
few pixels, line buffers the last few lines, frame stores the last few
frames. Frames are too large to keep on chip in general, so the store is
a hierarchy of two tiers:

- **near** -- on-chip lines of each past frame, the ones about to be used;
- **far** -- the whole past frames, where they fit: on chip, or external
  memory.

A transfer engine moves lines from far to near ahead of their use, in
raster order and in line-sized bursts, which is also what external memory
reads efficiently. The datapath never addresses memory: it reads the past
as lines the near tier delivers, and those are the line buffers a spatial
read across time needs anyway -- `v[t - 1, r + 1, c]` is the window rule
on that stream, with no copy between the transfer and the window. A
model that reads only the same pixel from each past frame needs no more
near tier than a few bursts in flight.

**A live stream cannot wait for memory.** Its source does not pause, so a
refill that arrives late is pixels lost, not a stall: the transfer runs
ahead by enough lines to cover the worst-case latency of a memory the
design shares with others, the bandwidth it needs leaves headroom, and the
blanking is where it catches up. These are requirements checked when the
design is built, not hoped for when it runs.

**The binding is the design's.** Block RAM and memory ports are shared by
everything a design holds, so where a store lives is chosen where the
design is composed -- on chip, the near tier the whole store; or external
memory, with the line-ahead transfer -- with a default, on chip when it
fits. The compiler states the requirement: the size, the bandwidth, and
the latency the stream can hide. A binding that cannot meet it is refused
by name at build. The datapath is the same whichever binding holds its
past.

### 7. The hardware level

**Nodes.** Pure: constants, ports, module parameters, register ports,
`add sub mul neg shr`, bit access and bit assembly, `clip abs`, comparisons,
one-bit `and or not`, `sel` -- a lookup on a small axis lowers to it or to
a constant table -- and `wrap`. Stateful: `reg` -- named, with a
reset value or none -- and `memory`, with depth, width, ports, read latency
and read-during-write, its reads and writes as nodes. Widths are explicit
and follow from ranges; no Verilog in any node.

**Counters come from variables.** Each driving variable of the source gets
a counter: the innermost advances on every accepted beat, each outer one
when the variable inside it wraps. First and last along a variable are its
counter at 0 and at count - 1; `sof`, `eol` and `last` are these.

**The output trails the input.** The output's driving variables are
counters of their own, behind the input's by the lag the window rule
derives -- for a padded 3x3 stencil, one row and one column. They advance
on an accepted beat, or on a FLUSH cycle: once the input's frame has ended,
an idle input cycle -- blanking, where a real source has it -- or, when the
next frame arrives first, a cycle in which `in_ready` holds it back. The
output's flags come from the output's counters.

**One ready.** Every stage register is enabled while the output is not
stalled; the stream's backpressure is the only handshake.

**Resets.** A register needs a reset only if its value is read without a
reset flag saying it is valid; data registers qualified by `out_valid` have
none.

### Left open, by design

Hierarchy by instances of named subgraphs; several clocks; resource-limited
schedules; a parser for dumps; pad modes beyond replicate and zero; frame
store bindings beyond on chip and a line-ahead transfer from external
memory (caches, compression).
