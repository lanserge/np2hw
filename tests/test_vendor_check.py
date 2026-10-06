# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A described vendor block against the pins its tool built."""
from np2hw.toolchain.vivado.vendor_check import compare, dumped

DUMP = """CELL rx
PIN RefClk I   clk
PIN aRst I   undef
PIN pRst I   undef
PIN pLocked O   undef
PIN vid_pData O 23 0 undef
"""
WANT = {"rx": {"block": "lib:hdmi_rx", "tie": ["pRst"],
               "pins": {"RefClk": ["I", 1], "aRst": ["I", 1],
                        "vid_pData": ["O", 24]}}}


def test_the_dump_reads_directions_and_widths():
    assert dumped(DUMP) == {"rx": {"RefClk": ["I", 1], "aRst": ["I", 1],
                                   "pRst": ["I", 1], "pLocked": ["O", 1],
                                   "vid_pData": ["O", 24]}}


def test_a_true_description_agrees_and_an_open_output_is_allowed():
    assert compare(WANT, dumped(DUMP)) == []


def test_a_wrong_width_a_missing_pin_and_an_unreached_input_are_named():
    want = {"rx": dict(WANT["rx"], tie=[],
                       pins={"RefClk": ["I", 1], "aRst": ["I", 1],
                             "vid_pData": ["O", 16], "vid_pDE": ["O", 1]})}
    problems = compare(want, dumped(DUMP))
    assert any("vid_pData as O 16, and it is O 24" in p for p in problems)
    assert any("describes vid_pDE, which the implementation does not have"
               in p for p in problems)
    assert any("pRst is an input nothing in the description reaches" in p
               for p in problems)


def test_tying_an_output_is_refused():
    want = {"rx": dict(WANT["rx"], tie=["pRst", "pLocked"])}
    assert any("ties pLocked, which is not an input" in p
               for p in compare(want, dumped(DUMP)))
