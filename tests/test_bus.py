# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The register bus: a decoder by the allocator's windows, a 2:1 arbiter,
an AXI3 bridge and a reset synchroniser -- each simulated against strict
bus models with random stalls (``tests/tb_bus.py``)."""
import shutil
from pathlib import Path

import pytest

from np2hw import hdl
from np2hw.bus import allocate, axil_decoder, windows

TESTS = Path(__file__).parent
simulated = pytest.mark.skipif(shutil.which("verilator") is None,
                               reason="verilator not on PATH")


def _run(tmp_path, verilog, top, case, parameters=None):
    pytest.importorskip("cocotb")
    from np2hw.testing import run_cocotb
    run_cocotb(tmp_path, verilog, top, "tb_bus", dict(case, seed=7),
               test_dir=TESTS, parameters=parameters)


GP0 = (0x4000_0000, 0x4000_0000)     # a processor's register region
SIZES = [0x8000, 0x1000, 0x1_0000]


@simulated
def test_the_decoder_routes_by_window(tmp_path):
    result = axil_decoder(SIZES, *GP0)
    windows = [result["facts"]["windows"][f"m{i:02d}_axi"]
               for i in range(len(SIZES))]
    top = max(b + s for b, s in windows)
    _run(tmp_path, hdl.closure(result), result["module"],
         {"dut": "decoder", "windows": windows, "count": 120,
          "unmapped": [0x0, top, top + 0x1000, 0x7FFF_FFFC]})


@simulated
def test_the_arbiter_serves_both_masters(tmp_path):
    _run(tmp_path, hdl.text("np2hw_axil_arbiter"), "np2hw_axil_arbiter",
         {"dut": "arbiter", "count": 24})


@simulated
def test_the_bridge_turns_bursts_into_beats(tmp_path):
    _run(tmp_path, hdl.text("np2hw_axi3_to_axil"), "np2hw_axi3_to_axil",
         {"dut": "bridge"})


@simulated
def test_the_reset_asserts_at_once_and_releases_on_the_clock(tmp_path):
    _run(tmp_path, hdl.text("np2hw_reset_sync"), "np2hw_reset_sync",
         {"dut": "reset", "stages": 3}, parameters={"STAGES": 3})


def test_a_window_a_bit_slice_cannot_decode_is_refused():
    with pytest.raises(ValueError, match="not a power of two"):
        windows([0x1000], [0x1800])
    with pytest.raises(ValueError, match="not aligned"):
        windows([0x1800], [0x1000])
    with pytest.raises(ValueError, match="overlap"):
        windows([0x0, 0x1000], [0x2000, 0x1000])


def test_decoders_of_different_maps_are_different_modules():
    a = axil_decoder([0x1000, 0x1000], *GP0)
    b = axil_decoder([0x1000, 0x2000], *GP0)
    assert a["module"] != b["module"]
    assert a["module"] == axil_decoder([0x1000, 0x1000], *GP0)["module"]


def test_windows_are_placed_largest_first_each_on_its_own_alignment():
    bases = allocate([0x1000, 0x8000, 0x1_0000, 0x1000], *GP0)
    assert bases == [0x4001_8000, 0x4001_0000, 0x4000_0000, 0x4001_9000]
    windows(bases, [0x1000, 0x8000, 0x1_0000, 0x1000])   # decodable
    assert allocate([0x1000, 0x8000, 0x1_0000, 0x1000], *GP0) == bases


def test_a_size_of_zero_is_a_port_not_made():
    """An optional slave absent from this build: its port is not made, the
    others keep their numbers, and it has no base."""
    result = axil_decoder([0x1000, 0, 0x2000], *GP0)
    names = {p["name"].split("_")[0] for p in result["interface"]["ports"]}
    assert {"m00", "m02"} <= names and "m01" not in names
    assert result["facts"]["windows"] == {"m00_axi": [0x4000_2000, 0x1000],
                                          "m02_axi": [0x4000_0000, 0x2000]}


def test_windows_that_do_not_fit_the_region_are_refused():
    with pytest.raises(ValueError, match="do not fit"):
        allocate([0x1000, 0x1000], 0x4000_0000, 0x1000)
