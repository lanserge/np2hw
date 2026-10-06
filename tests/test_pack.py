# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw.pipeline.pack: a pipeline's design pack, and the generator that
writes one for FuseSoC."""
import json
import shutil

import pytest

from np2hw import fusesoc, pipeline
from np2hw.pipeline import compose, pack
from np2hw.pipeline import run as runner

DESIGN = "toylib.pipelines"


def test_the_pack_is_a_direct_build_with_its_manifest(tmp_path):
    written = pack.emit(DESIGN, tmp_path, verify=False)
    direct = compose.generate(pipeline.load(DESIGN), control=True)
    assert written["verilog"].read_text() == direct.verilog
    manifest = written["core"].read_text()
    for artifact in ("verilog", "regmap", "systemrdl", "docs"):
        assert f"- {written[artifact].name}" in manifest
    assert f"toplevel: {direct.top}" in manifest
    # no vendor, the models' package, the pipeline's name
    assert "name: :toylib:pipelines:0" in manifest
    build = json.loads(written["build"].read_text())
    assert build["geometry"] == {"width": 64, "height": 8}


def test_control_false_stops_at_the_datapath(tmp_path):
    flat = pack.emit(DESIGN, tmp_path, control=False, verify=False)
    assert "s_axi" not in flat["verilog"].read_text()


def test_the_twin_context_moves_only_what_a_frame_can_stand_for():
    # phase may be set by a register or the stream: moved off its reset;
    # limit is a register-only choice and width follows the frame
    assert runner.synthetic_context(pipeline.load(DESIGN)) == {"phase": 2}


@pytest.mark.skipif(shutil.which("verilator") is None,
                    reason="verilator not on PATH")
def test_the_pack_is_verified_by_default(tmp_path, capsys):
    pack.emit(DESIGN, tmp_path)
    assert "twin bit-exact with the model" in capsys.readouterr().out


def test_a_diverging_twin_refuses_the_whole_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "run_rtl",
        lambda chain, frame, values, context, bit_depth: frame ^ 1)
    out = tmp_path / "pack"
    with pytest.raises(ValueError, match="DIFFERS"):
        pack.emit(DESIGN, out)
    assert not out.exists(), "a refused pack still wrote artifacts"


def _gapi(tmp_path, parameters, vlnv="acme:toy:pack:0"):
    path = tmp_path / "input.yml"
    path.write_text(json.dumps({"gapi": "1.0", "vlnv": vlnv,
                                "files_root": str(tmp_path),
                                "parameters": parameters}))
    return str(path)


def test_the_generator_writes_a_pipeline_pack(tmp_path, monkeypatch):
    gapi = _gapi(tmp_path, {"design": DESIGN, "verify": False})
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert fusesoc.main([gapi]) == 0
    (core,) = work.glob("*.core")
    assert "name: acme:toy:pack:0" in core.read_text()


def test_the_generator_needs_a_design_or_a_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="design"):
        fusesoc.main([_gapi(tmp_path, {})])


def test_a_verified_pack_keeps_its_vlnv(tmp_path, monkeypatch):
    # the twin is the model here: what is under test is that verifying
    # every input's chain leaves the manifest's name alone
    monkeypatch.setattr(
        runner, "run_rtl",
        lambda chain, frame, values, context, bit_depth:
            runner.run_model(chain, frame, values, context, bit_depth))
    written = pack.emit(DESIGN, tmp_path, vlnv="acme:toy:p:0")
    assert written["core"].name == "p.core"
    assert "name: acme:toy:p:0" in written["core"].read_text()


def test_a_pack_without_line_buffers_has_no_memories_to_bind(tmp_path):
    written = pack.emit(DESIGN, tmp_path, verify=False, sync_memory=True)
    assert json.loads(written["build"].read_text())["memories"] == {}
