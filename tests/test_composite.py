# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The JSON composition loader: references, settings, expressions, flattening.

Each test writes a small block library into a fresh package on the import
path, so references resolve exactly as they do for any library -- through
importlib, from the top of the import path.
"""
import itertools
import json

import pytest
from np2hw.pipeline import compose

from np2hw.design.composite import (CompositionError, evaluate, find, load,
                                      parse, references)

WB = "toylib.blocks:balance"
BL = "toylib.blocks:bias"
_count = itertools.count()


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Write {module: {block: {...}}} as a package; return a function that
    takes the files and gives back the package name."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def make(files, python: dict | None = None) -> str:
        """``files`` is {module: {block: {...}}}, or a function of the
        package name returning it, so a file can reference its own library."""
        name = f"fixture_lib_{next(_count)}"
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("")
        if callable(files):
            files = files(name)
        for module, content in files.items():
            # {block: {...}}, or a whole library file (blocks, profiles)
            whole = set(content) & {"blocks", "profiles"} == set(content) \
                and content
            (package / f"{module}.json").write_text(
                json.dumps(content if whole else {"blocks": content}))
        for module, text in (python or {}).items():
            (package / f"{module}.py").write_text(text)
        return name
    return make


def chain(*names_and_refs, **extra):
    """A composite: in -> each node in order -> out."""
    nodes = {n: {"block": r} for n, r in names_and_refs}
    order = [n for n, _ in names_and_refs]
    connections = [["in", f"{order[0]}.in"]]
    connections += [[f"{a}.out", f"{b}.in"] for a, b in zip(order, order[1:])]
    connections += [[f"{order[-1]}.out", "out"]]
    return dict({"inputs": [{"name": "in"}], "outputs": [{"name": "out"}],
                 "nodes": nodes, "connections": connections}, **extra)


# -- the toy ISP ------------------------------------------------------------ #

ISP = "toylib.isp:isp"
GEOMETRY = {"bit_depth": 12, "max_width": 64, "max_height": 32}

def test_an_unset_interface_parameter_is_refused():
    with pytest.raises(CompositionError, match="bit_depth is never set"):
        load(ISP)


# -- hierarchy -------------------------------------------------------------- #

def test_nested_composites_flatten_to_edges_between_leaves(library):
    lib = library(lambda lib: {
        "inner": {"pair": chain(("bl", BL), ("wb", WB))},
        "outer": {"top": chain(("a", f"{lib}.inner:pair"),
                               ("b", f"{lib}.inner:pair"))},
    })
    design = load(f"{lib}.outer:top")
    assert list(design.instances) == ["a.bl", "a.wb", "b.bl", "b.wb"]
    assert design.edges == [
        ("in", "a.bl.in"), ("a.bl.out", "a.wb.in"),
        ("a.wb.out", "b.bl.in"), ("b.bl.out", "b.wb.in"),
        ("b.wb.out", "out")]
    assert design.composites == {"": f"{lib}.outer:top",
                                 "a": f"{lib}.inner:pair",
                                 "b": f"{lib}.inner:pair"}


def test_a_composite_that_contains_itself_is_refused(library):
    lib = library(lambda lib: {"m": {"loop": {
        "nodes": {"x": {"block": f"{lib}.m:loop"}}}}})
    with pytest.raises(CompositionError, match="contains itself"):
        load(f"{lib}.m:loop")


def test_a_module_that_is_both_python_and_json_is_refused(library):
    lib = library({"both": {"b": chain(("wb", WB))}},
                  python={"both": "x = 1\n"})
    with pytest.raises(CompositionError, match="both a Python module"):
        find(f"{lib}.both:b")


# -- settings --------------------------------------------------------------- #

def test_the_enclosing_composite_has_the_last_word(library):
    inner = chain(("wb", WB))
    inner["nodes"]["wb"]["values"] = {"gain": [[300, 300], [300, 300]]}
    def files(lib):
        outer = chain(("isp", f"{lib}.inner:lib"))
        outer["nodes"]["isp"]["values"] = {"wb.gain": [[400, 256], [256, 400]]}
        return {"inner": {"lib": inner}, "outer": {"design": outer}}
    lib = library(files)
    design = load(f"{lib}.outer:design")
    assert design.instances["isp.wb"].values == {
        "gain": [[400, 256], [256, 400]]}
    # ...and without the outer setting, the inner one stands.
    assert load(f"{lib}.inner:lib").instances["wb"].values == {
        "gain": [[300, 300], [300, 300]]}


def test_a_setting_of_something_that_does_not_exist_is_refused(library):
    typo = chain(("wb", WB))
    typo["nodes"]["wb"]["values"] = {"gian": [[256] * 2] * 2}
    lib = library({"m": {"d": typo}})
    with pytest.raises(CompositionError,
                       match=r"no register 'gian' \(registers \['gain'\]\)"):
        load(f"{lib}.m:d")


def test_a_register_under_build_is_refused(library):
    wrong = chain(("wb", WB))
    wrong["nodes"]["wb"]["build"] = {"gain": 1}
    lib = library({"m": {"d": wrong}})
    with pytest.raises(CompositionError, match="set it under values"):
        load(f"{lib}.m:d")


def test_a_value_outside_its_register_is_refused(library):
    big = chain(("wb", WB))
    big["nodes"]["wb"]["values"] = {"gain": [[70000, 256], [256, 256]]}
    lib = library({"m": {"d": big}})
    with pytest.raises(CompositionError, match="70000 does not fit gain"):
        load(f"{lib}.m:d")


def test_a_value_of_the_wrong_shape_is_refused(library):
    flat = chain(("wb", WB))
    flat["nodes"]["wb"]["values"] = {"gain": 256}
    lib = library({"m": {"d": flat}})
    with pytest.raises(CompositionError, match=r"shape \(2, 2\), given \(\)"):
        load(f"{lib}.m:d")


def test_a_register_attribute_is_a_build_setting_and_configures_a_variant(
        library):
    d = chain(("wb", WB))
    d["nodes"]["wb"]["build"] = {"gain.frac": 12}
    lib = library({"m": {"d": d}})
    wb = load(f"{lib}.m:d").instances["wb"]
    assert wb.build == {"gain.frac": 12}
    assert next(p for p in wb.leaf.params.params if p.name == "gain").frac == 12


# -- expressions ------------------------------------------------------------ #

def test_build_expressions_reach_down_and_pass_parameters_in(library):
    inner = chain(("wb", WB), parameters={"depth": {}, "frac": {}})
    inner["nodes"]["wb"]["build"] = {"gain.frac": "frac"}
    def files(lib):
        outer = chain(("isp", f"{lib}.inner:lib"),
                      parameters={"depth": {"default": 12}})
        outer["nodes"]["isp"]["build"] = {"depth": "depth",
                                          "frac": "isp.depth - 2"}
        return {"inner": {"lib": inner}, "outer": {"design": outer}}
    lib = library(files)
    design = load(f"{lib}.outer:design")
    assert design.parameters == {"depth": 12}
    assert design.instances["isp.wb"].build == {"gain.frac": 10}
    assert load(f"{lib}.outer:design", build={"depth": 14}) \
        .instances["isp.wb"].build == {"gain.frac": 12}


def test_a_cycle_is_refused(library):
    inner = chain(("wb", WB), parameters={"a": {}, "b": {}})
    def files(lib):
        outer = chain(("isp", f"{lib}.inner:lib"))
        outer["nodes"]["isp"]["build"] = {"a": "isp.b + 1", "b": "isp.a"}
        return {"inner": {"lib": inner}, "outer": {"design": outer}}
    lib = library(files)
    with pytest.raises(CompositionError, match="a cycle"):
        load(f"{lib}.outer:design")


def test_one_register_alone_is_an_alias(library):
    d = chain(("left", WB), ("right", WB))
    d["nodes"]["left"]["values"] = {"gain": [[300, 256], [256, 300]]}
    d["nodes"]["right"]["values"] = {"gain": "left.gain"}
    lib = library({"m": {"d": d}})
    design = load(f"{lib}.m:d")
    assert design.aliases == {"right.gain": "left.gain"}
    assert "gain" not in design.instances["right"].values


def test_arithmetic_between_registers_is_refused(library):
    d = chain(("left", WB), ("right", WB))
    d["nodes"]["right"]["values"] = {"gain": "left.gain * 2"}
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match="computes with registers"):
        load(f"{lib}.m:d")


def test_a_build_parameter_cannot_follow_a_register(library):
    inner = chain(("wb", WB), parameters={"n": {}})
    def files(lib):
        outer = chain(("isp", f"{lib}.inner:lib"), ("wb", WB))
        outer["nodes"]["isp"]["build"] = {"n": "wb.gain"}
        return {"inner": {"lib": inner}, "outer": {"design": outer}}
    lib = library(files)
    with pytest.raises(CompositionError, match="reads wb.gain, a register"):
        load(f"{lib}.outer:design")


def test_a_register_value_from_build_parameters_is_a_constant(library):
    d = chain(("o", "toylib.blocks:offset"), ("g", "toylib.blocks:gain"),
              parameters={"n": {"default": 50}})
    d["nodes"]["o"]["values"] = {"offset": "n * 2"}
    d["nodes"]["g"]["values"] = {"gain": "(1 << 4) - n // 4"}
    lib = library({"m": {"d": d}})
    design = load(f"{lib}.m:d")
    assert design.instances["o"].values == {"offset": 100}
    assert design.instances["g"].values == {"gain": 4}


@pytest.mark.parametrize("text, message", [
    ("1 / 2", "true division"),
    ("1.5", "a constant is an integer"),
    ("__import__('os')", "Call is not allowed"),
    ("1 +", "is not an expression"),
])
def test_the_walker_accepts_only_integer_arithmetic(text, message):
    with pytest.raises(CompositionError, match=message):
        parse(text, "here")


def test_the_walker_computes_as_python_does_on_integers():
    names = {"a": 7, "b.c": -3}
    for text in ("a // 2", "b.c // 2", "a % 3", "b.c % 3", "a << 3",
                 "max(a, b.c) + min(a, 1) * abs(b.c)",
                 "a if b.c < 0 else 0", "(a > 2) + (a == 7)",
                 "~a ^ (a | 8) & 12"):
        tree = parse(text, "here")
        expected = eval(text.replace("b.c", "bc"), {}, {"a": 7, "bc": -3})
        assert evaluate(tree, names.__getitem__) == int(expected), text
    assert references(parse("x.y + max(z, 1)", "here")) == ["x.y", "z"]


# -- connections ------------------------------------------------------------ #

def test_an_unknown_port_is_refused(library):
    d = chain(("wb", WB))
    d["connections"][0] = ["in", "wb.input"]
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match=r"no input 'input'"):
        load(f"{lib}.m:d")


def test_a_sink_driven_twice_is_refused(library):
    d = chain(("a", WB), ("b", WB))
    d["connections"].append(["in", "b.in"])
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match="already driven"):
        load(f"{lib}.m:d")


def test_an_open_composite_output_is_refused(library):
    inner = chain(("wb", WB))
    inner["connections"] = inner["connections"][:-1]
    lib = library({"m": {"lib": inner}})
    with pytest.raises(CompositionError, match="output out is driven by "
                                               "nothing inside it"):
        load(f"{lib}.m:lib")


def test_an_undriven_composite_input_is_refused(library):
    def files(lib):
        outer = chain(("isp", f"{lib}.inner:lib"))
        outer["connections"] = outer["connections"][1:]
        return {"inner": {"lib": chain(("wb", WB))},
                "outer": {"design": outer}}
    lib = library(files)
    with pytest.raises(CompositionError, match="used inside it and driven "
                                               "by nothing"):
        load(f"{lib}.outer:design")


# -- the file format -------------------------------------------------------- #

def test_an_address_is_refused_by_the_schema(library):
    d = chain(("wb", WB))
    d["addresses"] = {"wb": 256}
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match="addresses"):
        load(f"{lib}.m:d")


def test_a_reserved_name_is_refused_by_the_schema(library):
    lib = library({"m": {"d": chain(("max", WB))}})
    with pytest.raises(CompositionError):
        load(f"{lib}.m:d")


def test_a_generator_takes_its_build_parameters_from_its_signature(library):
    d = {"nodes": {"hist": {"block": "np2hw.frame_history:frame_history",
                            "build": {"frames": 3, "bits": 12,
                                      "width": 64, "height": 32}}}}
    lib = library({"m": {"d": d}})
    hist = load(f"{lib}.m:d").instances["hist"]
    assert hist.kind == "generator"
    assert hist.build == {"frames": 3, "bits": 12, "width": 64, "height": 32}
    d["nodes"]["hist"]["build"]["max_width"] = 64
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match="not 'max_width'"):
        load(f"{lib}.m:d")


# -- the stream entering, and the bridge to today's build ------------------- #

def test_the_stream_is_stated_where_it_enters():
    design = load(ISP, build=GEOMETRY)
    assert design.streams == {"in": {
        "bits": 12, "max_width": 64, "max_height": 32,
        "context": ["width", "height", "phase"]}}


def test_a_wrapper_inherits_the_stream_and_states_nothing(library):
    def files(lib):
        outer = chain(("isp", ISP), ("g", "toylib.blocks:gain"))
        outer["nodes"]["isp"]["build"] = dict(GEOMETRY)
        return {"m": {"d": outer}}
    lib = library(files)
    design = load(f"{lib}.m:d")
    assert design.streams["in"]["bits"] == 12
    assert list(design.instances)[:2] == ["isp.bias", "isp.balance"]


def test_two_ports_stating_one_stream_differently_are_refused(library):
    def files(lib):
        outer = chain(("isp", ISP))
        outer["inputs"][0]["bits"] = 10
        outer["nodes"]["isp"]["build"] = dict(GEOMETRY)
        return {"m": {"d": outer}}
    lib = library(files)
    with pytest.raises(CompositionError, match="bits = 12, but .* states 10"):
        load(f"{lib}.m:d")


def test_a_stream_fact_on_an_output_is_refused(library):
    d = chain(("wb", WB))
    d["outputs"][0]["bits"] = 12
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match="traced, not declared"):
        load(f"{lib}.m:d")


def _built(design):
    """A loaded design, checked for generation as designs.load checks it:
    model blocks, the stream entering, a graph np2hw accepts."""
    compose.shape(design)
    compose.netlist(design)
    return design


def test_a_design_is_built_around_the_stream_entering():
    pipeline = _built(load(ISP, build=GEOMETRY))
    assert (pipeline.name, compose.shape(pipeline).spec.bit_depth, compose.shape(pipeline).width,
            compose.shape(pipeline).height) == ("isp", 12, 64, 32)
    assert [s.path for s in compose.stages(pipeline)[1:]] == [
        "bias", "balance"]
    assert compose.shape(pipeline).context == {"width", "height", "phase"}


def test_a_stream_that_states_no_geometry_is_refused(library):
    lib = library({"m": {"d": chain(("wb", WB))}})
    with pytest.raises(CompositionError, match="states no bits, max_width"):
        _built(load(f"{lib}.m:d"))


def test_what_is_not_an_isp_block_is_refused(library):
    d = chain(("wb", WB))
    d["nodes"]["hist"] = {"block": "np2hw.frame_history:frame_history",
                          "build": {"frames": 1, "bits": 12, "width": 8,
                                    "height": 8}}
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match=r"hist \(generator\)"):
        _built(load(f"{lib}.m:d"))


def test_a_designs_values_are_the_register_files_reset_values(library):
    """One place, three readers: the register map, the model's defaults and
    the generated register file's reset all carry what the design set."""
    d = chain(("wb", WB), ("o", "toylib.blocks:offset"))
    d["inputs"][0].update(bits=12, max_width=64, max_height=32)
    d["nodes"]["wb"]["values"] = {"gain": [[300, 256], [256, 301]]}
    d["nodes"]["o"]["values"] = {"offset": 7}
    lib = library({"m": {"d": d}})
    pipeline = _built(load(f"{lib}.m:d"))
    regmap = compose.register_map(pipeline)
    default = {f"{b['path']}.{r['name']}": r["default"]
               for b in regmap["blocks"] for r in b["registers"]}
    assert default["wb.gain_0_0"] == 300 and default["wb.gain_1_1"] == 301
    assert default["o.offset"] == 7
    assert compose.stage(pipeline, "wb").paramset.defaults()["gain_1_1"] == 301
    verilog = compose.generate(pipeline, control=True).verilog
    assert "reg_wb_gain_0_0<=300;" in verilog
    assert "reg_o_offset<=7;" in verilog


def test_undeclared_values_keep_their_declared_reset(library):
    d = chain(("wb", WB))
    d["inputs"][0].update(bits=12, max_width=64, max_height=32)
    lib = library({"m": {"d": d}})
    regmap = compose.register_map(_built(load(f"{lib}.m:d")))
    gain = {r["name"]: r["default"] for b in regmap["blocks"]
            if b["path"] == "wb" for r in b["registers"]}
    assert set(gain.values()) == {256}          # unity, as declared


def test_a_register_that_follows_another_is_refused(library):
    d = chain(("left", WB), ("right", WB))
    d["inputs"][0].update(bits=12, max_width=64, max_height=32)
    d["nodes"]["right"]["values"] = {"gain": "left.gain"}
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match="sharing one register"):
        _built(load(f"{lib}.m:d"))


# -- profiles --------------------------------------------------------------- #

from np2hw.design import profile as profiles  # noqa: E402

GAIN = [[300, 256], [256, 300]]


def _pair():
    """Two toy ISPs side by side."""
    return {"inputs": [{"name": "left"}, {"name": "right"}],
            "outputs": [{"name": "l_out"}, {"name": "r_out"}],
            "nodes": {"l": {"block": ISP, "build": dict(GEOMETRY)},
                      "r": {"block": ISP, "build": dict(GEOMETRY)}},
            "connections": [["left", "l.in"], ["right", "r.in"],
                            ["l.out", "l_out"], ["r.out", "r_out"]]}


def _lib(library, profiles_):
    """A library: the pair as a design, and the profiles given."""
    return library(lambda lib: {"design": {"pair": _pair()},
                                "tuning": {"profiles": profiles_(lib)}})


def test_a_profile_is_keyed_relative_to_its_block_and_lands_at_every_instance(
        library):
    lib = _lib(library, lambda lib: {"sensor": {
        "for": ISP, "values": {"balance.gain": GAIN}}})
    regmap = compose.register_map(_built(load(f"{lib}.design:pair")))
    assert {c["path"] for c in regmap["composites"]} == {"", "l", "r"}
    flat = profiles.flatten(profiles.resolve(f"{lib}.tuning:sensor"), regmap)
    assert flat == {f"{side}.balance.gain_{i}_{j}": GAIN[i][j]
                    for side in ("l", "r") for i in range(2)
                    for j in range(2)}


def test_an_included_profile_lands_at_its_node_and_the_outer_value_wins(
        library):
    lib = _lib(library, lambda lib: {
        "sensor": {"for": ISP, "values": {"balance.gain": GAIN,
                                          "bias.bias": [4, -4]}},
        "rig": {"for": f"{lib}.design:pair",
                "values": {"l": f"{lib}.tuning:sensor",
                           "l.balance.gain": [[400, 256], [256, 400]]}}})
    rig = profiles.resolve(f"{lib}.tuning:rig")
    assert rig.values["l.balance.gain"] == [[400, 256], [256, 400]]
    assert rig.sources["l.bias.bias"] == f"{lib}.tuning:sensor"
    assert "r.bias.bias" not in rig.values


def test_an_include_for_a_different_block_is_refused(library):
    lib = _lib(library, lambda lib: {
        "wb": {"for": WB, "values": {"gain": GAIN}},
        "rig": {"for": f"{lib}.design:pair", "values": {"l": f"{lib}.tuning:wb"}}})
    with pytest.raises(CompositionError, match="is for .*balance.*but l "
                                               "is .*isp"):
        profiles.resolve(f"{lib}.tuning:rig")


def test_a_register_the_block_does_not_have_is_refused(library):
    lib = _lib(library, lambda lib: {"sensor": {
        "for": ISP, "values": {"bias.biases": [0, 0]}}})
    with pytest.raises(CompositionError, match=r"no register 'biases'"):
        profiles.resolve(f"{lib}.tuning:sensor")


def test_a_profile_key_that_is_not_a_register_is_refused(library):
    """A misspelt key would otherwise set nothing and say nothing."""
    lib = _lib(library, lambda lib: {"sensor": {
        "for": ISP, "values": {"balance.gian": GAIN}}})
    with pytest.raises(CompositionError, match="has no register 'gian'"):
        profiles.resolve(f"{lib}.tuning:sensor")
    lib = _lib(library, lambda lib: {"sensor": {
        "for": ISP, "values": {"balanse.gain": GAIN}}})
    with pytest.raises(CompositionError, match="has no node 'balanse'"):
        profiles.resolve(f"{lib}.tuning:sensor")


def test_a_profile_including_itself_is_refused(library):
    lib = _lib(library, lambda lib: {"loop": {
        "for": f"{lib}.design:pair", "values": {"l": f"{lib}.tuning:loop"}}})
    with pytest.raises(CompositionError):
        profiles.resolve(f"{lib}.tuning:loop")


def test_the_built_map_checks_shape_and_range(library):
    lib = _lib(library, lambda lib: {
        "flat": {"for": ISP, "values": {"balance.gain": [256, 256]}},
        "big": {"for": ISP, "values": {"balance.gain":
                                       [[70000, 256], [256, 256]]}}})
    regmap = compose.register_map(_built(load(f"{lib}.design:pair")))
    with pytest.raises(CompositionError, match="shape does not match"):
        profiles.flatten(profiles.resolve(f"{lib}.tuning:flat"), regmap)
    with pytest.raises(CompositionError, match="70000 does not fit"):
        profiles.flatten(profiles.resolve(f"{lib}.tuning:big"), regmap)


def test_a_design_without_the_profiles_block_is_refused(library):
    lib = _lib(library, lambda lib: {"sensor": {
        "for": ISP, "values": {"balance.gain": GAIN}}})
    flat_design = compose.register_map(_built(load(f"{lib}.design:pair")))
    flat_design["composites"] = []
    with pytest.raises(CompositionError, match="has no toylib.isp"):
        profiles.flatten(profiles.resolve(f"{lib}.tuning:sensor"), flat_design)


# -- libraries outside any installed package -------------------------------- #

def _outside(tmp_path):
    """A library a user keeps anywhere: plain folders, no __init__.py,
    referring to toylib's blocks and to each other by reference."""
    root = tmp_path / "their_libs"
    (root / "acme" / "cams").mkdir(parents=True)
    (root / "acme" / "cams" / "imx999.json").write_text(json.dumps({
        "profiles": {"outdoor": {
            "for": ISP, "values": {"balance.gain": GAIN}}}}))
    (root / "acme" / "rig.json").write_text(json.dumps({
        "blocks": {"small": {
            "inputs": [{"name": "in"}], "outputs": [{"name": "out"}],
            "nodes": {"isp": {"block": ISP, "build": dict(GEOMETRY)}},
            "connections": [["in", "isp.in"], ["isp.out", "out"]]}},
        "profiles": {"field": {
            "for": "acme.rig:small",
            "values": {"isp": "acme.cams.imx999:outdoor"}}}}))
    return root


def test_a_library_directory_makes_outside_references_resolve(tmp_path):
    import os
    import subprocess
    import sys as _sys
    from pathlib import Path
    root = _outside(tmp_path)
    # toylib is on the import path as an installed library would be
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent)}
    out = tmp_path / "pack"
    ok = subprocess.run(
        [_sys.executable, "-m", "np2hw", "pipeline", "generate",
         "acme.rig:small", "--library", str(root), "--out", str(out), "--no-verify"],
        capture_output=True, text=True, cwd=tmp_path, env=env)
    assert ok.returncode == 0, ok.stderr
    regmap = out / "small_regmap.json"
    flat = subprocess.run(
        [_sys.executable, "-m", "np2hw.design.profile", "acme.rig:field",
         "--library", str(root), "--map", str(regmap)],
        capture_output=True, text=True, cwd=tmp_path, env=env)
    assert flat.returncode == 0, flat.stderr
    assert json.loads(flat.stdout) == {
        f"isp.balance.gain_{i}_{j}": GAIN[i][j]
        for i in range(2) for j in range(2)}
    # ...and without it, the reference means nothing here: refused, named.
    missing = subprocess.run(
        [_sys.executable, "-m", "np2hw.design.profile", "acme.rig:field",
         "--map", str(regmap)], capture_output=True, text=True, cwd=tmp_path,
        env=env)
    assert missing.returncode != 0 and "acme" in missing.stderr


def test_a_library_that_is_not_a_directory_is_refused(tmp_path):
    from np2hw.design.composite import add_libraries
    with pytest.raises(CompositionError, match="not a directory"):
        add_libraries([tmp_path / "nowhere"])


def test_a_reset_value_outside_its_register_is_refused_at_the_block():
    from toylib.blocks import balance
    with pytest.raises(ValueError, match="reset value 70000 outside"):
        balance.configure(resets={"gain": [[70000, 256], [256, 256]]})


# -- the shorthand ---------------------------------------------------------- #

def test_a_module_alone_names_the_block_called_after_its_file():
    from np2hw.design.composite import canonical
    assert find("toylib.isp").ref == "toylib.isp:isp"
    assert find("np2hw.frame_history").obj is \
        find("np2hw.frame_history:frame_history").obj
    assert canonical("toylib.blocks:gain") == "toylib.blocks:gain"
    with pytest.raises(CompositionError, match="has no 'blocks'"):
        find("toylib.blocks")


def test_short_and_full_references_are_one_block_everywhere(library):
    """A profile written for the short form anchors where the map records
    the full one: references are compared in one form."""
    lib = library(lambda lib: {
        "design": {"pair": _pair()},
        "tuning": {"profiles": {"s": {
            "for": "toylib.isp",
            "values": {"balance.gain": GAIN}}}}})
    regmap = compose.register_map(_built(load(f"{lib}.design:pair")))
    flat = profiles.flatten(profiles.resolve(f"{lib}.tuning:s"), regmap)
    assert len(flat) == 8


# -- designs for a toolchain: when, pins, vendor settings ------------------- #

IP = "toylib.ip"


def test_a_node_whose_when_is_false_is_not_built(library):
    d = {"parameters": {"extra": {"default": 0}},
         "nodes": {"a": {"block": f"{IP}:constant"},
                   "b": {"block": f"{IP}:constant", "when": "extra"}},
         "connections": []}
    lib = library({"m": {"d": d}})
    assert list(load(f"{lib}.m:d").instances) == ["a"]
    assert list(load(f"{lib}.m:d", build={"extra": 1}).instances) == ["a", "b"]


def test_a_connection_to_a_node_not_built_must_carry_the_same_when(library):
    d = {"parameters": {"extra": {"default": 0}},
         "nodes": {"a": {"block": f"{IP}:constant"},
                   "b": {"block": f"{IP}:logic", "when": "extra"}},
         "connections": [["a.dout", "b.Op1"]]}
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match="guard the connection"):
        load(f"{lib}.m:d")
    d["connections"] = [{"from": "a.dout", "to": "b.Op1", "when": "extra"}]
    lib = library({"m": {"d": d}})
    assert load(f"{lib}.m:d").edges == []
    assert load(f"{lib}.m:d", build={"extra": 1}).edges == [("a.dout", "b.Op1")]


def test_pins_are_gathered_and_are_the_fpgas_so_their_names_are_global(library):
    jack = {"pins": [{"name": "led", "dir": "O"}],
            "nodes": {"k": {"block": f"{IP}:constant"}},
            "connections": [["k.dout", "led"]]}
    def files(lib):
        top = {"nodes": {"j1": {"block": f"{lib}.jack:jack"},
                         "j2": {"block": f"{lib}.jack:jack"}},
               "connections": []}
        return {"jack": {"jack": jack}, "top": {"top": top}}
    lib = library(files)
    with pytest.raises(CompositionError, match="pin led is also"):
        load(f"{lib}.top:top")
    one = library(lambda lib: {"jack": {"jack": jack}, "top": {"top": {
        "nodes": {"j": {"block": f"{lib}.jack:jack"}}, "connections": []}}})
    design = load(f"{one}.top:top")
    assert design.pins == {"led": {"dir": "O"}}
    assert design.edges == [("j.k.dout", "pin:led")]


def test_vendor_settings_are_open_and_config_is_verbatim(library):
    d = {"parameters": {"n": {"default": 3}},
         "nodes": {"c": {"block": f"{IP}:concat", "build": {"NUM_PORTS": "n + 1"},
                         "config": {"IN0_WIDTH": "No_buffer"}}},
         "connections": []}
    lib = library({"m": {"d": d}})
    c = load(f"{lib}.m:d").instances["c"]
    assert c.build == {"NUM_PORTS": 4} and c.config == {"IN0_WIDTH": "No_buffer"}


def test_config_on_a_leaf_that_is_not_vendor_ip_is_refused(library):
    d = chain(("wb", WB))
    d["nodes"]["wb"]["config"] = {"x": 1}
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match="not vendor IP"):
        load(f"{lib}.m:d")


def test_a_default_may_be_derived_and_a_word_is_quoted(library):
    d = {"parameters": {"khz": {"default": 74250},
                        "genlock": {"default": "1 if khz < 100000 else 0"},
                        "clock": {"default": "'/a/b' if genlock else '/c/d'"}},
         "nodes": {"k": {"block": f"{IP}:constant"}}, "connections": []}
    lib = library({"m": {"d": d}})
    assert load(f"{lib}.m:d").parameters == {
        "khz": 74250, "genlock": 1, "clock": "/a/b"}
    assert load(f"{lib}.m:d", build={"khz": 148500}).parameters["clock"] == "/c/d"


def test_a_name_that_is_both_a_parameter_and_a_node_is_refused(library):
    d = {"parameters": {"k": {"default": 1}},
         "nodes": {"k": {"block": f"{IP}:constant"}}, "connections": []}
    lib = library({"m": {"d": d}})
    with pytest.raises(CompositionError, match="both a parameter and a node"):
        load(f"{lib}.m:d")


# -- composites a function writes ------------------------------------------- #

GEN = '''
def pair(config):
    ports = [{"name": n, **({"when": "extra"} if n == config["optional"] else {})}
             for n in config["names"]]
    nodes = {f"k_{n}": dict({"block": "toylib.ip:constant"},
                            **({"when": "extra"} if n == config["optional"] else {}))
             for n in config["names"]}
    conns = [dict({"from": f"k_{n}.dout", "to": n},
                  **({"when": "extra"} if n == config["optional"] else {}))
             for n in config["names"]]
    return {"parameters": {"extra": {"description": "x"}},
            "outputs": ports, "nodes": nodes, "connections": conns}
pair.composite = True

def broken(config):
    return {"nodes": {}}
broken.composite = True
'''


def test_a_function_can_write_a_composite_from_its_config(library):
    lib = library(lambda lib: {"m": {"d": {
        "parameters": {"extra": {"default": 0}},
        "nodes": {"g": {"block": f"{lib}.gen:pair", "build": {"extra": "extra"},
                        "config": {"names": ["a", "b"], "optional": "b"}},
                  "sink": {"block": f"{IP}:concat"}},
        "connections": [["g.a", "sink.In0"],
                        {"from": "g.b", "to": "sink.In1", "when": "extra"}]}}},
        python={"gen": GEN})
    d = load(f"{lib}.m:d")
    assert sorted(d.instances) == ["g.k_a", "sink"]
    assert d.edges == [("g.k_a.dout", "sink.In0")]
    d = load(f"{lib}.m:d", build={"extra": 1})
    assert sorted(d.instances) == ["g.k_a", "g.k_b", "sink"]


def test_a_written_composite_is_checked_like_one_written_by_hand(library):
    lib = library(lambda lib: {"m": {"d": {
        "nodes": {"g": {"block": f"{lib}.gen:broken"}}, "connections": []}}},
        python={"gen": GEN})
    with pytest.raises(CompositionError, match="wrote a composite that is not one"):
        load(f"{lib}.m:d")


# -- a hand-written leaf is wired by the ports it declares ------------------ #

ADAPTER = {"inputs": [{"name": "clk"},
                      {"name": "in", "width": 8, "stream": ["sof"]}],
           "outputs": [{"name": "out", "width": 8,
                        "stream": ["sof", "eol", "last"]},
                       {"name": "busy"}],
           "vivado": {"module": "adapter"}, "sources": ["adapter.v"]}


def _wired(library, *connections):
    lib = library(lambda name: {"leaves": {"adapter": ADAPTER}, "top": {
        "top": {"nodes": {"a": {"block": f"{name}.leaves:adapter"},
                          "b": {"block": f"{name}.leaves:adapter"}},
                "connections": [list(c) for c in connections]}}})
    return load(f"{lib}.top:top")


def test_a_declared_stream_is_wired_by_its_name(library):
    design = _wired(library, ("a.out", "b.in"))
    assert ("a.out", "b.in") in design.edges


def test_a_stream_wired_against_its_side_is_refused(library):
    with pytest.raises(CompositionError, match="has no output 'in'"):
        _wired(library, ("a.in", "b.in"))


def test_a_signal_of_a_stream_may_be_tapped(library):
    """A status word reads a valid off a stream it does not consume."""
    design = _wired(library, ("a.out", "b.in"), ("a.in_valid", "b.clk"))
    assert ("a.in_valid", "b.clk") in design.edges


def test_an_undeclared_port_of_a_declaring_leaf_is_refused(library):
    with pytest.raises(CompositionError, match="has no input 'enable'"):
        _wired(library, ("a.out", "b.enable"))


# -- a generator that makes a leaf ------------------------------------------ #

LEAF_GEN = '''
def slice(bits: int) -> dict:
    """A register slice, generated for its width."""
    from np2hw.ports import Pin, header, published
    from np2hw.stream import stream_pins
    pins = [Pin("clk", "input"), Pin("rst", "input"),
            *stream_pins("s", "in", bits), *stream_pins("m", "out", bits)]
    text = "\\n".join(header("slice", pins)) + "\\nendmodule\\n"
    return {"verilog": text, "module": "slice",
            "interface": {"ports": published(pins)}}


slice.leaf = True
'''


def _sliced(library, a_bits, b_bits, *connections):
    lib = library(lambda name: {"top": {"top": {
        "nodes": {"a": {"block": f"{name}.gen:slice",
                        "build": {"bits": a_bits}},
                  "b": {"block": f"{name}.gen:slice",
                        "build": {"bits": b_bits}}},
        "connections": [list(c) for c in connections]}}},
        python={"gen": LEAF_GEN})
    return f"{lib}.top:top"


def test_a_leaf_generator_is_wired_by_the_ports_it_publishes(library):
    design = load(_sliced(library, 8, 8, ("a.m", "b.s")))
    leaf = design.instances["a"].leaf
    assert leaf["vivado"]["module"] == "slice"
    assert "m_data" in {p["name"] for p in leaf["ports"]}
    assert ("a.m", "b.s") in design.edges
    # its arguments made the module; none is a parameter of it
    assert design.instances["a"].build == {}


def test_a_published_stream_wired_against_its_side_is_refused(library):
    with pytest.raises(CompositionError, match="has no output 's'"):
        load(_sliced(library, 8, 8, ("a.s", "b.s")))


def test_two_leaves_generating_one_file_differently_are_refused(library,
                                                                tmp_path):
    from np2hw.design.composite import write_generated
    design = load(_sliced(library, 8, 9))
    with pytest.raises(CompositionError, match="both generate slice.v"):
        write_generated(design, tmp_path / "hdl")


def test_reading_parameters_need_not_run_a_generator(library):
    ref = _sliced(library, 0, 0, ("a.m", "b.s"))
    with pytest.raises(CompositionError, match="a .*slice"):
        load(ref)
    assert load(ref, generate=False).instances["a"].kind == "generator"


# -- every wire against the ports it joins ---------------------------------- #

WIDE = {"parameters": {"W": {"description": "data width"}},
        "inputs": [{"name": "clk"}, {"name": "in", "width": "W",
                                     "stream": ["sof"]},
                   {"name": "level", "width": 4},
                   {"name": "s_axi", "width": 8, "bus": "axilite"},
                   {"name": "s_axis", "width": 32, "bus": "axis",
                    "signals": ["tlast"]}],
        "outputs": [{"name": "out", "width": "W", "stream": ["sof"]},
                    {"name": "m_axis", "width": "W", "bus": "axis",
                     "signals": ["tlast"]},
                    {"name": "count", "width": 8},
                    {"name": "flag"}],
        "vivado": {"module": "wide"}, "sources": ["wide.v"]}


def _wide(library, a_w, b_w, *connections):
    lib = library(lambda name: {"leaves": {"wide": WIDE}, "top": {"top": {
        "nodes": {"a": {"block": f"{name}.leaves:wide",
                        "build": {"W": a_w}},
                  "b": {"block": f"{name}.leaves:wide",
                        "build": {"W": b_w}}},
        "connections": [list(c) for c in connections]}}})
    return f"{lib}.top:top"


def test_a_stream_of_one_width_at_both_ends_is_wired(library):
    load(_wide(library, 12, 12, ("a.out", "b.in")))


def test_a_stream_whose_widths_differ_is_refused(library):
    with pytest.raises(CompositionError, match=r"a.out \(12 bits\) -> "
                       r"b.in \(10 bits\): WIDTH MISMATCH"):
        load(_wide(library, 12, 10, ("a.out", "b.in")))


def test_a_net_whose_pins_differ_in_width_is_refused(library):
    with pytest.raises(CompositionError, match="WIDTH MISMATCH on one net"):
        load(_wide(library, 8, 8, ("a.count", "b.level")))


def test_a_net_two_leaves_drive_is_refused(library):
    """Two outputs on one net is a short, however the net was reached --
    here through a tap of b's clock that also lands on b's own flag."""
    with pytest.raises(CompositionError, match="one net driven by"):
        load(_wide(library, 8, 8, ("a.flag", "b.clk"), ("b.clk", "b.flag")))


def test_a_stream_wired_to_a_bus_is_refused(library):
    with pytest.raises(CompositionError, match="not one kind of bus"):
        load(_wide(library, 8, 8, ("a.out", "b.s_axi")))


def test_a_width_reading_an_unset_parameter_is_refused(library):
    lib = library(lambda name: {"leaves": {"wide": WIDE}, "top": {"top": {
        "nodes": {"a": {"block": f"{name}.leaves:wide"},
                  "b": {"block": f"{name}.leaves:wide", "build": {"W": 8}}},
        "connections": [["a.out", "b.in"]]}}})
    with pytest.raises(CompositionError, match="reads W, which this instance "
                       "never sets"):
        load(f"{lib}.top:top")


def test_an_axi_stream_whose_data_widths_differ_is_refused(library):
    """The xRGB word into a raster built for the cable's 24 bits."""
    load(_wide(library, 32, 8, ("a.m_axis", "b.s_axis")))
    with pytest.raises(CompositionError, match=r"a.m_axis \(24 bits\) -> "
                       r"b.s_axis \(32 bits\): WIDTH MISMATCH"):
        load(_wide(library, 24, 8, ("a.m_axis", "b.s_axis")))


# -- which block fills a node: a parameter, set like any other -------------- #

CCM = "toylib.blocks:offset"
STATS = "toylib.blocks:meter"


def _chooser(name, default=f"@{BL}"):
    """a -> b, where b is whatever the parameter `second` names."""
    return dict(chain(("a", BL), ("b", BL)),
                parameters={"second": {"default": default}},
                nodes={"a": {"block": BL}, "b": {"block": "second"}})


def _filled(library, entry, **pair):
    """`pair` used by `top` as node `isp`, `entry` merged into that node."""
    lib = library(lambda name: {"lib": {
        "pair": _chooser(name, **pair),
        "top": dict(chain(("isp", f"{name}.lib:pair")),
                    nodes={"isp": {"block": f"{name}.lib:pair", **entry}})}})
    return load(f"{lib}.lib:top")


def test_a_node_may_be_filled_by_a_parameter(library):
    design = _filled(library, {})
    assert design.instances["isp.b"].ref == BL


def test_the_block_a_parameter_names_is_set_from_outside(library):
    design = _filled(library, {"build": {"second": f"@{WB}"}})
    assert design.instances["isp.b"].ref == WB
    assert design.instances["isp.a"].ref == BL


def test_the_outermost_setting_chooses(library):
    lib = library(lambda name: {"lib": {
        "pair": _chooser(name),
        "mid": dict(chain(("p", f"{name}.lib:pair")),
                    nodes={"p": {"block": f"{name}.lib:pair",
                                 "build": {"second": f"@{WB}"}}}),
        "top": dict(chain(("m", f"{name}.lib:mid")),
                    nodes={"m": {"block": f"{name}.lib:mid",
                                 "build": {"p.second": f"@{CCM}"}}})}})
    assert load(f"{lib}.lib:mid").instances["p.b"].ref == WB
    assert load(f"{lib}.lib:top").instances["m.p.b"].ref == CCM


def test_a_reference_is_an_expression_like_any_other(library):
    """`@` marks a reference inside an expression, so a parameter can
    choose between blocks by arithmetic on others."""
    lib = library(lambda name: {"lib": {
        "pair": dict(_chooser(name), parameters={
            "fast": {"default": 0},
            "second": {"default": f"@{WB} if fast else @{BL}"}}),
        "top": dict(chain(("isp", f"{name}.lib:pair")),
                    nodes={"isp": {"block": f"{name}.lib:pair",
                                   "build": {"fast": 1}}})}})
    assert load(f"{lib}.lib:pair").instances["b"].ref == BL
    assert load(f"{lib}.lib:top").instances["isp.b"].ref == WB


def test_a_bare_name_that_is_no_parameter_is_refused(library):
    lib = library(lambda name: {"lib": {"pair": dict(
        _chooser(name), nodes={"a": {"block": BL}, "b": {"block": "nothing"}})}})
    with pytest.raises(CompositionError, match="neither a reference nor a "
                       "parameter"):
        load(f"{lib}.lib:pair")


def test_a_parameter_naming_no_block_is_refused(library):
    with pytest.raises(CompositionError, match="names no block"):
        _filled(library, {"build": {"second": 3}})


def test_a_block_with_other_ports_cannot_fill_a_node(library):
    """Everything around the node was wired to its default's ports."""
    with pytest.raises(CompositionError, match="cannot fill isp.second"):
        _filled(library, {"build": {"second": f"@{STATS}"}})


def test_a_setting_for_the_replaced_block_is_refused(library):
    with pytest.raises(CompositionError, match="has no register 'bias'"):
        _filled(library, {"build": {"second": f"@{WB}"},
                          "values": {"b.bias": [0, 0]}})


# -- a composite of model blocks inside a design, built whole --------------- #

SOURCE = {"parameters": {"W": {}}, "inputs": [{"name": "clk"}],
          "outputs": [{"name": "out", "width": "W",
                       "stream": ["sof", "eol", "last"]}],
          "vivado": {"module": "src"}, "sources": ["src.v"]}
SINK = {"parameters": {"W": {}},
        "inputs": [{"name": "in", "width": "W",
                    "stream": ["sof", "eol", "last"]}],
        "vivado": {"module": "dst"}, "sources": ["dst.v"]}


def _mixed(library, dst_width="out_bits", isp_build=None):
    """src -> isp -> dst, the ISP a composite of model blocks whose input
    states its stream; the top takes the depth in from the ISP's input and
    the word out from its trace, and wires both into its neighbours."""
    def files(name):
        isp = chain(("bl", BL), ("wb", WB))
        isp["parameters"] = {"w": {"default": 64}}
        isp["inputs"][0].update(bits=12, max_width="w", max_height=32)
        top = {"parameters": {"bits": {"default": "isp.in.bits"},
                              "out_bits": {"default": "isp.out.data_bits"}},
               "nodes": {"src": {"block": f"{name}.lib:source",
                                 "build": {"W": "bits"}},
                         "isp": {"block": f"{name}.lib:isp",
                                 "build": isp_build or {}},
                         "dst": {"block": f"{name}.lib:sink",
                                 "build": {"W": dst_width}}},
               "connections": [["src.out", "isp.in"], ["isp.out", "dst.in"]]}
        return {"lib": {"source": SOURCE, "sink": SINK, "isp": isp,
                        "top": top}}
    return f"{library(files)}.lib:top"


def test_a_composite_of_model_blocks_in_a_design_is_one_cell(library):
    design = load(_mixed(library))
    isp = design.instances["isp"]
    assert isp.kind == "primitive" and "isp.bl" not in design.instances
    names = {p["name"] for p in isp.leaf["ports"]}
    assert {"in_valid", "in_data", "out_valid", "s_axil_awaddr"} <= names
    assert ("src.out", "isp.in") in design.edges
    assert ("isp.out", "dst.in") in design.edges
    assert any(f.endswith("_regmap.json") for f in isp.leaf["files"])


def test_a_parameter_may_be_assigned_a_nodes_facts(library):
    """Declared where the stream enters, traced where it leaves."""
    design = load(_mixed(library))
    assert design.parameters["bits"] == 12
    assert design.parameters["out_bits"] == 12          # bias, balance: 12
    assert design.instances["dst"].build == {"W": 12}


def test_a_wire_disagreeing_with_the_traced_word_is_refused(library):
    with pytest.raises(CompositionError, match="WIDTH MISMATCH"):
        load(_mixed(library, dst_width=10))


def test_a_parameter_that_needs_its_own_trace_is_a_cycle(library):
    with pytest.raises(CompositionError, match="a cycle"):
        load(_mixed(library, isp_build={"w": "out_bits"}))


def test_a_design_of_model_blocks_alone_is_not_built_whole(library):
    lib = library({"lib": {"isp": dict(
        chain(("bl", BL)),
        inputs=[{"name": "in", "bits": 12, "max_width": 64,
                 "max_height": 32}])}})
    design = load(f"{lib}.lib:isp")
    assert design.instances["bl"].kind == "block"


# -- a block described apart from its implementations ----------------------- #

RX = {"parameters": {"implementation": {"default": "'a'"},
                     "speed": {"default": 2}},
      "inputs": [{"name": "clk"}],
      "outputs": [{"name": "q", "width": 8}],
      "implementations": {
          "a": {"vivado": {"ip": "vendor.com:ip:a"},
                "ports": {"clk": "CLK_A", "q": "Q_A"},
                "facts": {"order": [2, 0, 1]},
                "config": {"MODE": "'fast' if speed > 1 else 'slow'"}},
          "b": {"vivado": {"module": "bmod"}, "sources": ["b.v"],
                "facts": {"order": [0, 1, 2]}}}}
SNK = {"parameters": {"W": {}}, "inputs": [{"name": "d", "width": "W"}],
       "vivado": {"module": "snk"}, "sources": ["snk.v"]}


def _with_rx(library, rx_entry=None):
    lib = library(lambda name: {"lib": {
        "rx": RX, "snk": SNK,
        "top": {"parameters": {"first": {"default": "rx.order[0]"}},
                "nodes": {"rx": {"block": f"{name}.lib:rx", **(rx_entry or {})},
                          "snk": {"block": f"{name}.lib:snk",
                                  "build": {"W": 8}}},
                "connections": [["rx.q", "snk.d"]]}}})
    return load(f"{lib}.lib:top")


def test_a_block_is_filled_by_the_implementation_its_parameter_chooses(
        library):
    design = _with_rx(library)
    rx = design.instances["rx"]
    assert rx.leaf["vivado"] == {"ip": "vendor.com:ip:a"}
    assert rx.leaf["port_map"] == {"clk": "CLK_A", "q": "Q_A"}
    # the implementation's settings, evaluated over the block's parameters;
    # the block's parameters themselves are not the vendor's
    assert rx.config == {"MODE": "fast"}
    assert rx.build == {}
    # wired, and checked, by the block's own ports
    assert ("rx.q", "snk.d") in design.edges


def test_the_implementation_is_set_from_outside_and_its_facts_follow(
        library):
    design = _with_rx(library, {"build": {"implementation": "'b'"}})
    rx = design.instances["rx"]
    assert rx.leaf["vivado"] == {"module": "bmod"}
    assert rx.leaf["sources"] == ["b.v"]
    assert design.parameters["first"] == 0
    assert _with_rx(library).parameters["first"] == 2


def test_an_implementations_settings_follow_the_blocks_parameters(library):
    design = _with_rx(library, {"build": {"speed": 1}})
    assert design.instances["rx"].config == {"MODE": "slow"}


def test_an_implementation_the_block_lacks_is_refused(library):
    with pytest.raises(CompositionError, match=r"no implementation 'c' "
                       r"\(implementations \['a', 'b'\]\)"):
        _with_rx(library, {"build": {"implementation": "'c'"}})


def test_a_block_is_wired_by_its_own_port_names(library):
    with pytest.raises(CompositionError, match="Q_A"):
        lib = library(lambda name: {"lib": {
            "rx": RX, "snk": SNK,
            "top": {"nodes": {"rx": {"block": f"{name}.lib:rx"},
                              "snk": {"block": f"{name}.lib:snk",
                                      "build": {"W": 8}}},
                    "connections": [["rx.Q_A", "snk.d"]]}}})
        load(f"{lib}.lib:top")


def test_a_vendor_blocks_ports_are_checked_like_any_others(library):
    lib = library(lambda name: {"lib": {
        "rx": RX, "snk": SNK,
        "top": {"nodes": {"rx": {"block": f"{name}.lib:rx"},
                          "snk": {"block": f"{name}.lib:snk",
                                  "build": {"W": 4}}},
                "connections": [["rx.q", "snk.d"]]}}})
    with pytest.raises(CompositionError, match="rx.q.*8.*snk.d.*4|"
                                               "snk.d.*4.*rx.q.*8"):
        load(f"{lib}.lib:top")


def test_a_port_exists_only_when_the_blocks_parameters_say(library):
    clock = {"parameters": {"implementation": {"default": "'x'"},
                            "shift": {"default": 0}},
             "inputs": [{"name": "clk_in"},
                        {"name": "psen", "when": "shift"}],
             "outputs": [{"name": "clk_out"}],
             "implementations": {"x": {"vivado": {"ip": "v:x"}}}}

    def top(shift):
        return {"nodes": {"c": {"block": "{lib}.lib:clock",
                                "build": {"shift": shift}}},
                "connections": []}

    for shift, has in ((0, False), (1, True)):
        lib = library(lambda name, shift=shift: {"lib": {
            "clock": clock,
            "top": json.loads(json.dumps(top(shift)).replace("{lib}",
                                                              name))}})
        leaf = load(f"{lib}.lib:top").instances["c"].leaf
        assert ("psen" in [p["name"] for p in leaf["inputs"]]) is has
        assert not any("when" in p for p in leaf["inputs"])


# -- a design's expression may call a function its library offers ---------- #

FUNCTIONS = '''
def double(n):
    return 2 * n
double.expression = True

def hidden(n):
    return n
'''


def _calling(library, call):
    lib = library(lambda name: {"lib": {
        "top": dict(chain(("a", BL)),
                    parameters={"n": {"default": 21},
                                "m": {"default": call.replace("{lib}",
                                                              name)}})}},
        python={"fns": FUNCTIONS})
    return load(f"{lib}.lib:top")


def test_an_expression_calls_a_function_its_library_offers(library):
    assert _calling(library, "@{lib}.fns:double(n) + 1").parameters["m"] == 43


def test_a_function_that_does_not_offer_itself_is_refused(library):
    with pytest.raises(CompositionError,
                       match=r"hidden does not say it may be called"):
        _calling(library, "@{lib}.fns:hidden(n)")


def test_a_composite_input_that_does_not_exist_needs_no_driver(library):
    """A port with a `when` exists only where it is true: an input the
    variant does not have is not refused for being undriven."""
    def lib(flag):
        return library(lambda name: {"lib": {
            "inner": dict(chain(("a", BL)),
                          parameters={"extra": {"default": 0}},
                          inputs=[{"name": "in"},
                                  {"name": "aux", "when": "extra"}],
                          nodes={"a": {"block": BL},
                                 "b": {"block": BL, "when": "extra"}},
                          connections=[["in", "a.in"], ["a.out", "out"],
                                       {"from": "aux", "to": "b.in",
                                        "when": "extra"}]),
            "top": dict(chain(("i", f"{name}.lib:inner")),
                        nodes={"i": {"block": f"{name}.lib:inner",
                                     "build": {"extra": flag}}})}})
    assert "i.a" in load(f"{lib(0)}.lib:top").instances
    with pytest.raises(CompositionError, match="aux.*driven by nothing"):
        load(f"{lib(1)}.lib:top")
