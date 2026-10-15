# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""``np2hw pipeline``: a pipeline as a library of image functions."""
import subprocess
import sys

import numpy as np
import pytest

from np2hw import cli

DESIGN = "toylib.pipelines"


@pytest.fixture()
def frame(tmp_path):
    path = tmp_path / "in.npy"
    np.save(path, np.random.default_rng(3).integers(0, 100, size=(8, 64)))
    return path


def _run(*arguments) -> int:
    return cli.main(["pipeline", "run", *map(str, arguments)])


def test_split_at_a_port_equals_one_run(tmp_path, frame):
    assert _run(DESIGN, frame, tmp_path / "full.npy") == 0
    assert _run(DESIGN, "--to", "g.out", frame, tmp_path / "mid.npy") == 0
    assert _run(DESIGN, "--from", "o.in", tmp_path / "mid.npy",
                tmp_path / "rest.npy") == 0
    np.testing.assert_array_equal(np.load(tmp_path / "rest.npy"),
                                  np.load(tmp_path / "full.npy"))


def test_set_changes_the_output_and_is_range_checked(tmp_path, frame, capsys):
    assert _run(DESIGN, frame, tmp_path / "a.npy") == 0
    assert _run(DESIGN, "--set", "g.gain=3", frame, tmp_path / "b.npy") == 0
    a, b = np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy")
    assert not np.array_equal(a, b)
    assert _run(DESIGN, "--set", "g.gain=99", frame, tmp_path / "c.npy") == 2
    assert "np2hw pipeline run:" in capsys.readouterr().err


def test_a_context_the_pipeline_lacks_is_refused(tmp_path, frame, capsys):
    assert _run(DESIGN, "--context", "phase=4", frame, tmp_path / "x.npy") == 2
    assert "phase holds 0..3" in capsys.readouterr().err
    assert _run(DESIGN, "--context", "tilt=1", frame, tmp_path / "x.npy") == 2
    assert "has no 'tilt'" in capsys.readouterr().err


def test_generate_writes_the_pack(tmp_path):
    assert cli.main(["pipeline", "generate", DESIGN, "--no-verify",
                     "--out", str(tmp_path), "--vlnv", "acme:toy:p:0"]) == 0
    assert (tmp_path / "pipelines.v").exists()
    assert "name: acme:toy:p:0" in (tmp_path / "p.core").read_text()


def test_python_dash_m_is_the_command(tmp_path, frame):
    done = subprocess.run(
        [sys.executable, "-m", "np2hw", "pipeline", "run", DESIGN,
         str(frame), str(tmp_path / "out.npy")],
        capture_output=True, text=True,
        cwd=str(__import__("pathlib").Path(__file__).parent))
    assert done.returncode == 0, done.stderr


RGB = "toylib.pipelines:rgb"


@pytest.fixture()
def samples(tmp_path):
    path = tmp_path / "samples.npy"
    np.save(path, np.random.default_rng(7).integers(0, 1 << 8, size=(48, 64)))
    return path


def test_injection_is_just_math(tmp_path, samples):
    """An (h, w, 2) array injected at the joiner is accepted on the strength
    of its shape alone, as NumPy would accept it."""
    two = np.stack([np.load(samples)] * 2, axis=-1)
    np.save(tmp_path / "two.npy", two)
    assert _run(RGB, "--from", "t.in", tmp_path / "two.npy",
                tmp_path / "out.npy") == 0
    assert np.load(tmp_path / "out.npy").shape == (48, 64, 3)


def test_png_out_and_back(tmp_path, samples):
    from np2hw.pipeline.run import read_frame
    assert _run(RGB, samples, tmp_path / "out.png") == 0
    image = read_frame(tmp_path / "out.png", 10)
    assert image.shape == (48, 64, 3)
    # the PNG is the 8-bit view of the same frame
    assert _run(RGB, samples, tmp_path / "out.npy") == 0
    exact = np.load(tmp_path / "out.npy")
    assert np.array_equal(image >> 2, exact >> 2)


def test_two_channel_png_is_refused(tmp_path, samples, capsys):
    assert _run(RGB, "--to", "p.out", samples, tmp_path / "out.png") == 2
    assert "2-channel" in capsys.readouterr().err


def test_explain_names_every_layer(tmp_path, samples, capsys):
    assert _run(RGB, "--profile", "toylib.tuning:rgb", "--explain",
                "--set", "g.gain=3", samples, tmp_path / "out.npy") == 0
    output = capsys.readouterr().out
    assert "g.gain = 3  (cli)" in output
    assert "(default)" in output
    assert _run(RGB, "--profile", "toylib.tuning:rgb", "--explain",
                samples, tmp_path / "out.npy") == 0
    assert "g.gain = 1  (profile)" in capsys.readouterr().out


def test_a_profile_for_another_design_is_refused(tmp_path, samples, capsys):
    assert _run(RGB, "--profile", "toylib.tuning:camera", samples,
                tmp_path / "out.npy") == 2
    assert "has no toylib.isp" in capsys.readouterr().err


@pytest.mark.skipif(__import__("shutil").which("verilator") is None,
                    reason="verilator not on PATH")
@pytest.mark.parametrize("compiler", ["default", "reference"])
def test_a_long_chain_takes_each_frame_from_the_block_before(
        tmp_path, samples, capsys, monkeypatch, compiler):
    """Six blocks, each handed its frame size by the one before: the twin
    holds reset for four clocks, so a size register that started anywhere
    but the traced size would still be rippling when the frame arrives.
    On both compilers: each starts its registers in its own code."""
    if compiler == "reference":
        monkeypatch.setenv("NP2HW_CORE", "off")
    assert _run("toylib.pipelines:long", "--rtl", samples,
                tmp_path / "rtl.npy") == 0
    assert "bit-exact" in capsys.readouterr().out


def test_the_rtl_twin_is_bit_exact(tmp_path, samples, capsys):
    assert _run(RGB, "--rtl", samples, tmp_path / "rtl.npy") == 0
    assert "bit-exact" in capsys.readouterr().out
    assert _run(RGB, samples, tmp_path / "model.npy") == 0
    assert np.array_equal(np.load(tmp_path / "rtl.npy"),
                          np.load(tmp_path / "model.npy"))
