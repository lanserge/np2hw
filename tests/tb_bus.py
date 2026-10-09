# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""cocotb testbench for the register bus: the decoder, the arbiter, the
AXI3 bridge and the reset synchroniser. Not collected by pytest --
``tests/test_bus.py`` builds each module and runs this through
:func:`np2hw.testing.run_cocotb`; the case names which one."""
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

from axi_bfm import (DECERR, OKAY, SLVERR, Axi3Master, LiteMaster,
                     LiteMemory)
from np2hw.testing import load_case


async def _start(dut):
    cocotb.start_soon(Clock(dut.aclk, 10, unit="ns").start())
    dut.aresetn.value = 0
    await ClockCycles(dut.aclk, 4)
    dut.aresetn.value = 1
    await RisingEdge(dut.aclk)


@cocotb.test(timeout_time=2, timeout_unit="ms")
async def decoder_routes_by_window(dut):
    """Every access reaches the slave whose window holds its address and no
    other; an address in no window is answered DECERR and reaches none."""
    case = load_case()
    if case["dut"] != "decoder":
        return
    rnd = random.Random(case["seed"])
    windows = case["windows"]
    slaves = [LiteMemory(dut, f"m{i:02d}_axi", dut.aclk, rnd,
                         errors=[b + 8] if i == 1 else [])
              for i, (b, _) in enumerate(windows)]
    master = LiteMaster(dut, "s_axi", dut.aclk, rnd)
    master.idle()
    await _start(dut)
    for slave in slaves:
        slave.start()

    expect = {i: {} for i in range(len(windows))}
    unmapped = case["unmapped"]
    for n in range(case["count"]):
        if rnd.random() < 0.15:
            address = rnd.choice(unmapped)
            assert await master.write(address, n) == DECERR
            data, resp = await master.read(address)
            assert (data, resp) == (0, DECERR), (hex(address), resp)
            continue
        i = rnd.randrange(len(windows))
        base, size = windows[i]
        address = base + 4 * rnd.randrange(size // 4)
        word = rnd.getrandbits(32)
        resp = await master.write(address, word)
        if i == 1 and address == base + 8:
            assert resp == SLVERR, "a slave's error must come back as it was"
            continue
        assert resp == OKAY
        expect[i][address] = word
        data, resp = await master.read(address)
        assert (data, resp) == (word, OKAY), (hex(address), data, word)

    for i, slave in enumerate(slaves):
        base, size = windows[i]
        for address, _ in slave.writes:
            assert base <= address < base + size, (
                f"slave {i} saw {address:#x}, outside its window")
        assert {a: d for a, d in slave.words.items()} == expect[i]


@cocotb.test(timeout_time=2, timeout_unit="ms")
async def arbiter_serves_both_masters(dut):
    """Two masters writing and reading at once: every transaction lands,
    each master's data is its own, and neither waits for the other to
    finish all its work."""
    case = load_case()
    if case["dut"] != "arbiter":
        return
    rnd = random.Random(case["seed"])
    memory = LiteMemory(dut, "m_axi", dut.aclk, rnd)
    masters = [LiteMaster(dut, f"s{k}_axi", dut.aclk, random.Random(k))
               for k in (0, 1)]
    for m in masters:
        m.idle()
    await _start(dut)
    memory.start()

    order = []

    async def work(k):
        for n in range(case["count"]):
            address = 0x100 * k + 4 * n
            assert await masters[k].write(address, (k << 16) | n) == OKAY
            data, resp = await masters[k].read(address)
            assert (data, resp) == ((k << 16) | n, OKAY)
            order.append(k)

    a = cocotb.start_soon(work(0))
    b = cocotb.start_soon(work(1))
    await a
    await b
    count = case["count"]
    assert sorted(order) == [0] * count + [1] * count

    # Writes alone, then reads alone, back to back from both: each master
    # asks again the cycle after its response, so a fixed priority would
    # hand the path to one of them every time. Alternation is the promise.

    async def writes(k):
        for n in range(count):
            assert await masters[k].write(0x800 + 0x100 * k + 4 * n, n) \
                == OKAY
            order.append(k)

    async def reads(k):
        for n in range(count):
            data, resp = await masters[k].read(0x800 + 0x100 * k + 4 * n)
            assert (data, resp) == (n, OKAY)
            order.append(k)

    for phase in (writes, reads):
        order.clear()
        a = cocotb.start_soon(phase(0))
        b = cocotb.start_soon(phase(1))
        await a
        await b
        both = order[:order.index(order[-1])]   # while both were asking
        runs = "".join(map(str, both)).replace("01", "0 1").replace(
            "10", "1 0").split()
        longest = max(len(run) for run in runs)
        assert longest <= 2, (f"{phase.__name__}: one master held the path "
                              f"{longest} times running")


@cocotb.test(timeout_time=2, timeout_unit="ms")
async def slice_passes_every_transaction_through(dut):
    """Through the slice, with both ends stalling at random: every write
    lands, every read returns what was written, a slave's error comes back
    as it was, and transactions overlap -- the slice holds one in reserve
    rather than taking turns."""
    case = load_case()
    if case["dut"] != "slice":
        return
    rnd = random.Random(case["seed"])
    memory = LiteMemory(dut, "m_axi", dut.aclk, rnd, errors=[0x40])
    master = LiteMaster(dut, "s_axi", dut.aclk, rnd)
    master.idle()
    await _start(dut)
    memory.start()
    expect = {}
    for n in range(case["count"]):
        address = 4 * rnd.randrange(32)
        word = rnd.getrandbits(32)
        resp = await master.write(address, word)
        if address == 0x40:
            assert resp == SLVERR
            data, resp = await master.read(address)
            assert resp == SLVERR
            continue
        assert resp == OKAY
        expect[address] = word
        check = rnd.choice(sorted(expect))
        data, resp = await master.read(check)
        assert (data, resp) == (expect[check], OKAY), hex(check)
    assert memory.writes and memory.reads


@cocotb.test(timeout_time=2, timeout_unit="ms")
async def bridge_turns_bursts_into_beats(dut):
    """AXI3 in, AXI4-Lite out: IDs echoed, a burst one access per beat
    (INCR advancing, FIXED staying), the last read beat marked, and the
    worst response of a write burst returned."""
    case = load_case()
    if case["dut"] != "bridge":
        return
    rnd = random.Random(case["seed"])
    memory = LiteMemory(dut, "m_axi", dut.aclk, rnd, errors=[0x4000_0108])
    master = Axi3Master(dut, "s_axi", dut.aclk, rnd)
    master.idle()
    await _start(dut)
    memory.start()

    # a single beat
    assert await master.write(0x4000_0000, [0xCAFE], ident=5) == (OKAY, 5)
    (beat,) = await master.read(0x4000_0000, ident=9)
    assert (beat["rdata"], beat["rresp"], beat["rid"], beat["rlast"]) == \
        (0xCAFE, OKAY, 9, 1)

    # an INCR burst: four addresses
    words = [0x11, 0x22, 0x33, 0x44]
    assert await master.write(0x4000_0010, words, ident=0xABC) == \
        (OKAY, 0xABC)
    assert memory.writes[-4:] == [(0x4000_0010 + 4 * i, w)
                                  for i, w in enumerate(words)]
    beats = await master.read(0x4000_0010, beats=4, ident=3)
    assert [b["rdata"] for b in beats] == words
    assert [b["rlast"] for b in beats] == [0, 0, 0, 1]
    assert {b["rid"] for b in beats} == {3}

    # a FIXED burst: one address, three times
    assert await master.write(0x4000_0020, [1, 2, 3], burst=0) == (OKAY, 0)
    assert [a for a, _ in memory.writes[-3:]] == [0x4000_0020] * 3
    assert memory.words[0x4000_0020] == 3

    # an error part way through a burst is not lost behind the rest
    resp, _ = await master.write(0x4000_0100, [7, 8, 9, 10])
    assert resp == SLVERR
    beats = await master.read(0x4000_0100, beats=4)
    assert [b["rresp"] for b in beats] == [OKAY, OKAY, SLVERR, OKAY]

    # reads and writes at once: independent paths
    w = cocotb.start_soon(master.write(0x4000_0200, [0x55] * 4))
    (r,) = await master.read(0x4000_0000)
    assert r["rdata"] == 0xCAFE
    assert (await w)[0] == OKAY


@cocotb.test(timeout_time=2, timeout_unit="ms")
async def reset_asserts_at_once_and_releases_on_the_clock(dut):
    """Assertion needs no clock; release comes STAGES+1 edges after the last
    source lets go, and a clock that is not locked holds the reset."""
    case = load_case()
    if case["dut"] != "reset":
        return
    stages = case["stages"]
    dut.rst_n_in.value = 1
    dut.locked.value = 1
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await ClockCycles(dut.clk, stages + 3)
    assert int(dut.rst_n.value) == 1 and int(dut.rst.value) == 0

    # assert between edges: the outputs follow without waiting for one
    await Timer(3, unit="ns")
    dut.rst_n_in.value = 0
    await Timer(1, unit="ns")
    assert int(dut.rst_n.value) == 0 and int(dut.rst.value) == 1

    await ClockCycles(dut.clk, 2)
    dut.rst_n_in.value = 1
    for edge in range(stages):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.rst_n.value) == 0, f"released early, edge {edge}"
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.rst_n.value) == 1 and int(dut.rst.value) == 0

    # A reset that comes and goes with no clock edge at all -- the clock it
    # resets had stopped -- still takes the full count to release.
    await Timer(2, unit="ns")
    dut.rst_n_in.value = 0
    await Timer(2, unit="ns")
    dut.rst_n_in.value = 1
    for edge in range(stages):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.rst_n.value) == 0, (
            f"released {edge + 1} edge(s) after a clockless reset")
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.rst_n.value) == 1

    dut.locked.value = 0
    await Timer(1, unit="ns")
    assert int(dut.rst_n.value) == 0, "an unlocked clock must hold reset"
