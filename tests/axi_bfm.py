# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""AXI masters and a memory slave for the bus tests, by the letter of the
protocol: a valid drops on the edge its handshake happens, address and data
are independent channels, and every ready is random.

Every coroutine is entered just after a rising edge, so assignments land in
the cycle they mean; a handshake is sampled in ReadOnly, on the values the
next edge will register.
"""
import random

import cocotb
from cocotb.triggers import ReadOnly, RisingEdge

OKAY, EXOKAY, SLVERR, DECERR = 0, 1, 2, 3


class Port:
    """The signals of one bundle, ``<prefix>_<signal>``."""

    def __init__(self, dut, prefix, clk):
        self.dut, self.prefix, self.clk = dut, prefix, clk

    def __getitem__(self, name):
        return getattr(self.dut, f"{self.prefix}_{name}")


# A handshake that has not come in this many cycles never will: the test
# fails naming the channel, instead of waiting out the simulator.
LIMIT = 1000


async def offer(port, valid, ready, fields):
    """Drive a channel's payload and valid until its handshake; return on
    the edge after it, with valid low again."""
    for name, value in fields.items():
        port[name].value = value
    port[valid].value = 1
    for _ in range(LIMIT):
        await ReadOnly()
        taken = int(port[ready].value)
        await RisingEdge(port.clk)
        if taken:
            port[valid].value = 0
            return
    raise AssertionError(f"{port.prefix}_{valid}: no {ready} in {LIMIT} "
                         "cycles")


async def take(port, valid, ready, fields, rnd, p=0.5, limit=LIMIT):
    """Raise ready at random until a beat is taken; return its payload.
    A slave waiting for its next request passes ``limit=None``."""
    cycles = 0
    while limit is None or cycles < limit:
        port[ready].value = int(rnd.random() < p)
        await ReadOnly()
        if int(port[ready].value) and int(port[valid].value):
            got = {name: int(port[name].value) for name in fields}
            await RisingEdge(port.clk)
            port[ready].value = 0
            return got
        await RisingEdge(port.clk)
        cycles += 1
    raise AssertionError(f"{port.prefix}_{valid}: nothing in {limit} cycles")


class LiteMaster:
    """One AXI4-Lite transaction at a time per call."""

    def __init__(self, dut, prefix, clk, rnd):
        self.p, self.rnd = Port(dut, prefix, clk), rnd

    def idle(self):
        for name in ("awvalid", "wvalid", "bready", "arvalid", "rready"):
            self.p[name].value = 0

    async def write(self, address, data, strb=0xF):
        aw = cocotb.start_soon(offer(self.p, "awvalid", "awready",
                                     {"awaddr": address}))
        w = cocotb.start_soon(offer(self.p, "wvalid", "wready",
                                    {"wdata": data, "wstrb": strb}))
        await aw
        await w
        b = await take(self.p, "bvalid", "bready", ["bresp"], self.rnd)
        return b["bresp"]

    async def read(self, address):
        await offer(self.p, "arvalid", "arready", {"araddr": address})
        r = await take(self.p, "rvalid", "rready", ["rdata", "rresp"],
                       self.rnd)
        return r["rdata"], r["rresp"]


class Axi3Master:
    """AXI3 transactions with an ID, single beats or bursts."""

    def __init__(self, dut, prefix, clk, rnd):
        self.p, self.rnd = Port(dut, prefix, clk), rnd

    def idle(self):
        for name in ("awvalid", "wvalid", "bready", "arvalid", "rready"):
            self.p[name].value = 0
        for name in ("awlock", "awcache", "awprot", "awqos", "arlock",
                     "arcache", "arprot", "arqos", "wid"):
            self.p[name].value = 0

    async def write(self, address, words, ident=0, burst=1, size=2):
        aw = cocotb.start_soon(offer(self.p, "awvalid", "awready", {
            "awaddr": address, "awid": ident, "awlen": len(words) - 1,
            "awsize": size, "awburst": burst}))
        for i, word in enumerate(words):
            await offer(self.p, "wvalid", "wready", {
                "wdata": word, "wstrb": 0xF, "wid": ident,
                "wlast": int(i == len(words) - 1)})
        await aw
        b = await take(self.p, "bvalid", "bready", ["bresp", "bid"],
                       self.rnd)
        return b["bresp"], b["bid"]

    async def read(self, address, beats=1, ident=0, burst=1, size=2):
        await offer(self.p, "arvalid", "arready", {
            "araddr": address, "arid": ident, "arlen": beats - 1,
            "arsize": size, "arburst": burst})
        out = []
        for _ in range(beats):
            out.append(await take(self.p, "rvalid", "rready",
                                  ["rdata", "rresp", "rid", "rlast"],
                                  self.rnd))
        return out


class LiteMemory:
    """An AXI4-Lite slave over a dictionary of words, with random stalls.
    Addresses in `errors` answer SLVERR; every write is logged."""

    def __init__(self, dut, prefix, clk, rnd, errors=()):
        self.p, self.rnd = Port(dut, prefix, clk), rnd
        self.words, self.writes, self.reads = {}, [], []
        self.errors = set(errors)
        for name in ("awready", "wready", "bvalid", "arready", "rvalid"):
            self.p[name].value = 0

    def start(self):
        cocotb.start_soon(self._writes())
        cocotb.start_soon(self._reads())

    async def _writes(self):
        while True:
            aw = cocotb.start_soon(take(self.p, "awvalid", "awready",
                                        ["awaddr"], self.rnd, limit=None))
            w = cocotb.start_soon(take(self.p, "wvalid", "wready",
                                       ["wdata", "wstrb"], self.rnd,
                                       limit=None))
            address = (await aw)["awaddr"]
            data = (await w)["wdata"]
            for _ in range(self.rnd.randrange(3)):
                await RisingEdge(self.p.clk)
            error = address in self.errors
            if not error:
                self.words[address] = data
            self.writes.append((address, data))
            await offer(self.p, "bvalid", "bready",
                        {"bresp": SLVERR if error else OKAY})

    async def _reads(self):
        while True:
            address = (await take(self.p, "arvalid", "arready", ["araddr"],
                                  self.rnd, limit=None))["araddr"]
            for _ in range(self.rnd.randrange(3)):
                await RisingEdge(self.p.clk)
            error = address in self.errors
            self.reads.append(address)
            await offer(self.p, "rvalid", "rready", {
                "rdata": 0 if error else self.words.get(address, 0),
                "rresp": SLVERR if error else OKAY})
