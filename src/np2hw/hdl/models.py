# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""NumPy models of np2hw's fixed modules: what each one's Verilog does.

A fixed module is hand-written Verilog, not traced from a model, so a design
holding one cannot be proved from its models unless the module brings one.
These are those models. Most fixed modules pass data through unchanged and
need none beyond the identity; this file holds the ones whose output is not
their input.

A module's library description names its model (``"model"``), and the
model takes the module's parameters by the names the description gives
them, the same values its Verilog is built with: one statement of each.

A model is driven with the SAME stimulus as the Verilog: the frames, and
where a module's output depends on when things happen, the timing -- when
each input frame arrives, when the consumer takes each output frame. The
model computes from that alone what the hardware must put out; nothing is
read back from the hardware. A timed module's implementation states the one
fact of its hardware the model needs, as its ``latency`` fact: the most
cycles by which the Verilog may see an event later than the model's
timeline puts it. A stimulus keeps every pair of events the module must
tell apart further apart than that (`Timeline.close`), so the model and the
hardware never break a tie differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Stimulus:
    """When a stream's frames come and go, in clock cycles from reset.

    Input frame k starts ``gaps[k]`` cycles after the frame before it ended
    -- the first, ``gaps[0]`` cycles after reset -- and sends one beat a
    cycle, as a sensor does. The consumer starts taking output frame 0 at
    cycle ``first``, and output frame i, ``waits[i - 1]`` cycles after it took
    the last beat of frame i - 1. Inside a frame it holds off for ``cycles``
    at each ``(beat, cycles)`` of ``stalls[i]``.
    """
    gaps: tuple[int, ...]
    first: int
    waits: tuple[int, ...]
    stalls: tuple[tuple[tuple[int, int], ...], ...] = ()

    def stall(self, i: int) -> tuple[tuple[int, int], ...]:
        return self.stalls[i] if i < len(self.stalls) else ()


@dataclass
class Timeline:
    """The events of a run: when each input frame's first and last beats
    went in, and when each output frame was chosen, began and ended."""
    started: list[int] = field(default_factory=list)
    arrived: list[int] = field(default_factory=list)
    restarted: list[int] = field(default_factory=list)  # starts of a history
    overrun: bool = False               # a frame read was being overwritten
    warmed: list[bool] = field(default_factory=list)  # a choice the store's
                                        # becoming warm made, not the consumer
    chosen: list[int] = field(default_factory=list)
    began: list[int] = field(default_factory=list)
    ended: list[int] = field(default_factory=list)

    def close(self, latency: int, upto: int | None = None) -> bool:
        """Whether a choice -- of the first `upto` -- has an event within
        `latency` of it that the store must tell apart from it: a frame's
        arrival, other than the one that made the store warm and so caused
        the choice, or the start of a frame that starts a history; or a
        consumer beginning a frame within `latency` of its choice. A tie the
        model and the hardware might break differently."""
        chosen = self.chosen[:upto]
        if any(b - c <= latency for c, b in zip(chosen, self.began)):
            return True
        for c, warmed in zip(chosen, self.warmed):
            for a in self.restarted + self.arrived:
                if abs(a - c) <= latency and not (warmed and c == a + 1):
                    return True
        return False


@dataclass(frozen=True)
class FrameOut:
    """One frame the store put out: the input frame it was computed at, and
    the frames side by side, newest in channel 0."""
    source: int
    data: np.ndarray                    # (h, w, frames)
    context: object


def frame_store(frames, stimulus: Stimulus, params: dict, contexts=None,
                outputs: int | None = None):
    """np2hw.hdl.frames:frame_store: the newest FRAMES whole frames, side by
    side, at the moment each output frame is chosen, from a ring of SLOTS.

    Args:
        frames: the input frames in order, each ``(h, w)``; a frame of
            another size than the one before starts the history over.
        stimulus: when they arrive and when the consumer takes the output.
        params: the node's parameters, by the description's names: FRAMES
            and SLOTS are read.
        contexts: what rides with each input frame; each output frame
            carries its source's.
        outputs: how many output frames to compute; by default as many as
            the stimulus has waits for.

    Returns:
        ``(outputs, timeline)``: each :class:`FrameOut` in order, and the
        events, for a stimulus generator to check for ties. Where the
        source reached a frame being read -- the store's `overrun` --
        ``timeline.overrun`` says so, and the frames out from then on are
        not what the hardware puts out: their samples may be of two frames.
    """
    depth, slots = int(params["FRAMES"]), int(params["SLOTS"])
    frames = [np.asarray(f) for f in frames]
    contexts = list(contexts) if contexts is not None else [None] * len(frames)
    t = Timeline()
    # the input side: every frame's last beat, and the history it belongs to
    end, epoch, epochs, size = -1, -1, [], None
    for k, f in enumerate(frames):
        start = end + 1 + stimulus.gaps[k]
        end = start + f.size - 1
        t.started.append(start)
        t.arrived.append(end)
        if f.shape != size:
            epoch, size = epoch + 1, f.shape
            t.restarted.append(start)
        epochs.append(epoch)

    def warm(at: int) -> int | None:
        """The newest frame whole before cycle `at`, if the history in force
        then -- the newest frame STARTED's -- holds `depth` whole frames, or
        None: a frame of a new size starts a history as it starts."""
        begun = [k for k, s in enumerate(t.started) if s < at]
        if not begun:
            return None
        whole = [k for k in begun if t.arrived[k] < at
                 and epochs[k] == epochs[begun[-1]]]
        return whole[-1] if len(whole) >= depth else None

    def ready_at(after: int) -> int | None:
        """The first cycle at or after `after` at which the store is warm."""
        for c in sorted({after, *(a + 1 for a in t.arrived if a + 1 > after)}):
            if warm(c) is not None:
                return c
        return None

    out = []
    count = outputs if outputs is not None else len(stimulus.waits) + 1
    previous_end = None
    for i in range(count):
        after = 0 if previous_end is None else previous_end + 1
        chosen = ready_at(after)
        if chosen is None:
            break
        n = warm(chosen)
        t.chosen.append(chosen)
        t.warmed.append(chosen != after)
        if previous_end is not None and i - 1 >= len(stimulus.waits):
            break                       # chosen, but never taken
        began = (stimulus.first if previous_end is None
                 else previous_end + stimulus.waits[i - 1])
        h, w = frames[n].shape
        ended = began + h * w - 1 + sum(c for _, c in stimulus.stall(i))
        t.began.append(began)
        t.ended.append(ended)
        # overrun: a read beginning with frames begun past its newest that
        # reach its oldest's slot, or a frame beginning, while it is read,
        # in a slot it reads
        held = {(n - k) % slots for k in range(depth)}
        ahead = [m for m, st in enumerate(t.started) if m > n and st < chosen]
        if ahead and ahead[-1] - n > slots - depth:
            t.overrun = True
        if any(chosen <= st <= ended and m % slots in held
               for m, st in enumerate(t.started) if m > n):
            t.overrun = True
        data = np.stack([frames[n - k] for k in range(depth)], axis=-1)
        out.append(FrameOut(source=n, data=data, context=contexts[n]))
        previous_end = ended
    return out, t
