# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Load a block hierarchy from JSON: composites of composites, down to leaves.

Everything is a BLOCK, and every folder is a LIBRARY of blocks. A block is
named the way Python names an entry point, from the top of the import path:

    acme.blocks.ccm:ccm                    a leaf: a block model in ccm.py
    np2hw.frame_history:frame_history      a leaf: a structural generator
    acme.boards.zynq7.ip:axi_gpio          a primitive: a JSON block, no nodes
    acme.pipelines.isp:isp                 a composite: a JSON block with nodes

A module alone names the block called after its file: ``acme.blocks.ccm``
is ``acme.blocks.ccm:ccm``, the common case spelt once; the full form
names any block in a file that holds several.

A JSON file always has one shape, ``{"blocks": {name: {...}}}``
(blocks.schema.json). A composite has ports, nodes -- each an instance of a
block, by reference -- the connections between them, and SETTINGS for what
it contains, written on the node they set:

    "isp": {"block": "acme.pipelines.isp:isp",
            "build":  {"bit_depth": 12},          what changes the hardware
            "values": {"ccm.m": [[...], ...]}}    what the hardware holds

Keys are paths relative to the node. A setting written further out wins over
one written further in, so the design -- the block you build -- has the last
word over the libraries it uses:

    leaf declaration  <  composite  <  enclosing composite  <  ...  <  design

A number or an array is a literal. A string is ALWAYS an expression, in Python
syntax, over the parameters in scope of the composite it is written in: its
own parameters, and anything inside its nodes (``isp.bit_depth``). It is
parsed with ``ast`` and evaluated by one walker that knows integer arithmetic,
shifts, comparisons, ``a if c else b``, ``min``, ``max`` and ``abs`` -- never
by ``eval``. What an expression becomes depends on what it references:

    nothing, or build parameters only   an integer, at load
    one register, alone                 an ALIAS: two paths, one register
    registers in arithmetic             refused, until np2hw can trace
                                        arithmetic between two parameters

What load() returns is the tree FLATTENED: every leaf instance at its full
path (``isp.ccm``), with its build settings and register values resolved,
and every connection joined across composite boundaries into an edge between
two leaves. The hierarchy survives as paths, which are the same paths in the
register map, the host API and a control surface.

Every library is found the same way: by the
import path, never by a file path, so a reference means the same thing on
every machine. ``acme.cams.imx999:outdoor`` is ``acme/cams/imx999.json``
wherever ``acme`` is importable -- installed, on PYTHONPATH, or named for one
run with a tool's ``--library DIR``. The folders need no ``__init__.py``.

Refusals name the path. A setting of something that does not exist is a typo;
a value outside its register is a picture that would be wrong; a cycle has no
answer.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import keyword
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "blocks.schema.json"


class CompositionError(ValueError):
    """A hierarchy that cannot be built, and where."""


# -- references ------------------------------------------------------------- #

@dataclass(frozen=True)
class Found:
    """What a reference names."""
    ref: str
    kind: str          # "composite" | "primitive" | "profile" | "block" | "generator"
    obj: object        # the JSON dict, the Block, or the callable
    folder: Path       # where the file lives: a primitive's files are beside it


LIBRARY_HELP = ("a directory to find libraries in, for this run: a "
                "reference a.b:name then also looks for DIR/a/b.json. "
                "Repeatable; the first given is searched first")


def add_libraries(directories) -> None:
    """Put library directories on the import path, first given first.

    A reference never holds a file path -- it means the same thing on
    every machine -- so a library outside any installed package is made
    findable the one way everything is found: by the import path."""
    import sys
    for directory in reversed(list(directories or ())):
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            raise CompositionError(f"--library {directory}: not a directory")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    importlib.invalidate_caches()


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@lru_cache(maxsize=None)
def _library_file(path: Path) -> dict:
    import jsonschema
    data = json.loads(path.read_text())
    try:
        jsonschema.validate(instance=data, schema=schema())
    except jsonschema.ValidationError as error:
        where = "/".join(str(p) for p in error.absolute_path) or "(top)"
        raise CompositionError(f"{path}: at {where}: {error.message}") from None
    return data


def canonical(ref: str) -> str:
    """A reference in its full form. ``package.module`` alone names the
    block called after its file -- ``acme.pipelines.isp`` is
    ``acme.pipelines.isp:isp`` -- which is the common case, one
    block to a file, spelt once. Everything that compares references
    compares this form."""
    module, sep, name = ref.partition(":")
    if not sep:
        name = module.rpartition(".")[2]
    return f"{module}:{name}"


def block_name(ref: str) -> str:
    """The block's own name, after the colon of the full form."""
    return canonical(ref).rpartition(":")[2]


def find(ref: str) -> Found:
    """Resolve ``package.module:block`` -- or ``package.module``, the block
    named after its file -- through importlib, so a checkout and an
    installed wheel behave the same. A module that is both a Python file
    and a JSON file is refused: one name, one thing. The Found carries the
    full form."""
    ref = canonical(ref)
    module, sep, name = ref.partition(":")
    package, dot, leaf = module.rpartition(".")
    if not (dot and name):
        raise CompositionError(f"{ref!r} is not a block reference "
                               "(package.module[:block])")
    try:
        pkg = importlib.import_module(package)
    except ImportError as error:
        raise CompositionError(f"{ref}: cannot import {package}: {error}") \
            from None
    # A folder of data files need not be a Python package with an
    # __init__.py: a namespace package has a path and no file.
    folder = Path(next(iter(pkg.__path__))).resolve()
    source = folder / f"{leaf}.json"
    python = importlib.util.find_spec(module) is not None
    if source.is_file() and python:
        raise CompositionError(
            f"{ref}: {module} is both a Python module and {source.name}; "
            "a library holds one thing under one name")
    if source.is_file():
        library = _library_file(source)
        blocks = library.get("blocks", {})
        profiles = library.get("profiles", {})
        if name in blocks and name in profiles:
            raise CompositionError(f"{ref}: {source} names both a block and "
                                   f"a profile {name!r}; a name means one "
                                   "thing in a file")
        if name in profiles:
            return Found(ref, "profile", profiles[name], folder)
        if name not in blocks:
            raise CompositionError(f"{ref}: {source} defines "
                                   f"{sorted(blocks) + sorted(profiles)}, "
                                   f"not {name!r}")
        block = blocks[name]
        kind = "composite" if "nodes" in block else "primitive"
        return Found(ref, kind, block, folder)
    if not python:
        raise CompositionError(f"{ref}: no module {module} and no "
                               f"{source.name} in {folder}")
    obj = getattr(importlib.import_module(module), name, None)
    if obj is None:
        raise CompositionError(f"{ref}: {module} has no {name!r}")
    from np2hw.blocks import Block
    if isinstance(obj, Block):
        return Found(ref, "block", obj, folder)
    if callable(obj):
        return Found(ref, "generator", obj, folder)
    raise CompositionError(f"{ref}: {module}.{name} is neither a block nor "
                           "a generator")


# -- expressions ------------------------------------------------------------ #

_BINARY = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.FloorDiv: lambda a, b: a // b,
           ast.Mod: lambda a, b: a % b, ast.Pow: lambda a, b: a ** b,
           ast.LShift: lambda a, b: a << b, ast.RShift: lambda a, b: a >> b,
           ast.BitAnd: lambda a, b: a & b, ast.BitOr: lambda a, b: a | b,
           ast.BitXor: lambda a, b: a ^ b}
_UNARY = {ast.USub: lambda a: -a, ast.UAdd: lambda a: +a,
          ast.Invert: lambda a: ~a, ast.Not: lambda a: int(not a)}
_COMPARE = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
            ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}
_CALLS = {"min": min, "max": max, "abs": abs}
# Calls that read the design's wiring, bound per node once it is wired: an
# expression using one is evaluated then ({value, wired}), never at load.
_WIRED_CALLS = ("same_net",)


# `@package.module:block` in an expression is a reference to a block. It is
# not Python -- nor is the colon -- so each one becomes a string constant
# before the expression is parsed, and the walker sees an ordinary string.
_REFERENCE = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
                        r"(?::[A-Za-z_][A-Za-z0-9_]*)?)")


# A path may pass through a port called `in` -- `isp.in.bits` -- and Python
# reads `in` as an operator. A keyword AFTER a dot is a name in a path, so it
# is renamed before parsing and named back where the path is read.
_KEYWORD_STEP = re.compile(
    r"(?<=[A-Za-z0-9_])\.(" + "|".join(keyword.kwlist) + r")\b(?!\s*\()")
_KW = "_kw_"


def parse(text: str, where: str) -> ast.expr:
    """Parse an expression and check every node is one the walker knows. A
    string that is not an expression is refused here, at load, with its
    path -- not later, where it would look like something else."""
    try:
        text_py = _REFERENCE.sub(lambda m: repr(m.group(1)), text)
        text_py = _KEYWORD_STEP.sub(lambda m: f".{_KW}{m.group(1)}", text_py)
        tree = ast.parse(text_py, mode="eval").body
    except SyntaxError as error:
        raise CompositionError(f"{where}: {text!r} is not an expression: "
                               f"{error.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute, ast.Load, ast.IfExp,
                             ast.BoolOp, ast.And, ast.Or, ast.Subscript)):
            continue
        if isinstance(node, ast.Constant):
            # Integers, and strings for the vendor settings that are words
            # ('/ps7/FCLK_CLK0'); a register value is still checked against
            # its register, which no string fits.
            if type(node.value) not in (int, str):
                raise CompositionError(
                    f"{where}: {text!r}: {node.value!r} -- a constant is an "
                    "integer, or a string for a vendor setting")
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            continue
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            continue
        if isinstance(node, ast.Compare) and all(type(op) in _COMPARE
                                                 for op in node.ops):
            continue
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in (*_CALLS, *_WIRED_CALLS)
                and not node.keywords):
            continue
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Constant)
                and isinstance(node.func.value, str) and not node.keywords):
            continue                     # @package.module:function(...)
        if type(node) in _BINARY or type(node) in _UNARY or \
                type(node) in _COMPARE:
            continue                     # operator tokens inside the above
        what = ("true division; use // (exact integers only)"
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                else type(node).__name__)
        raise CompositionError(f"{where}: {text!r}: {what} is not allowed "
                               "in a parameter expression")
    return tree


def _dotted(node: ast.expr) -> str | None:
    """``a.b.c`` as a string, if the node is a parameter path."""
    parts = []
    while isinstance(node, ast.Attribute):
        attr = node.attr
        parts.append(attr[len(_KW):] if attr.startswith(_KW) else attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id not in (*_CALLS,
                                                      *_WIRED_CALLS):
        return ".".join([node.id, *reversed(parts)])
    return None


def references(tree: ast.expr) -> list[str]:
    """Every parameter path an expression reads, outermost chains only."""
    found, inner = [], set()
    for node in ast.walk(tree):
        if id(node) in inner:
            continue
        path = _dotted(node)
        if path is not None:
            found.append(path)
            inner.update(id(n) for n in ast.walk(node))
    return found


def expression_function(ref: str):
    """The function ``@package.module:function`` names in an expression.
    It must say it may be called from one (``fn.expression = True``): a
    design is data, and what it can run is what its library offers it --
    a table's lookup, say, never whatever happens to be importable."""
    module, colon, name = ref.partition(":")
    if not colon:
        raise CompositionError(f"@{ref} is called, and a call names a "
                               "function: @package.module:function")
    try:
        fn = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise CompositionError(f"@{ref}: {error}") from None
    if not getattr(fn, "expression", False):
        raise CompositionError(f"@{ref} does not say it may be called from "
                               "an expression (expression = True)")
    return fn


def evaluate(tree: ast.expr, lookup, wired=None) -> int:
    """The one walker. ``lookup(path)`` binds each parameter path; with ints
    bound the result is an int, exactly as the model would compute it.
    ``wired`` binds the calls that read the wiring, where it is known."""
    def walk(node):
        path = _dotted(node)
        if path is not None:
            return lookup(path)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](walk(node.operand))
        if isinstance(node, ast.Compare):
            left = walk(node.left)
            for op, right in zip(node.ops, node.comparators):
                right = walk(right)
                if not _COMPARE[type(op)](left, right):
                    return 0
                left = right
            return 1
        if isinstance(node, ast.BoolOp):
            values = [walk(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return int(all(values))
            return int(any(values))
        if isinstance(node, ast.IfExp):
            return walk(node.body) if walk(node.test) else walk(node.orelse)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Constant):
            return expression_function(node.func.value)(
                *(walk(a) for a in node.args))
        if isinstance(node, ast.Call) and node.func.id in _WIRED_CALLS:
            if wired is None:
                raise CompositionError(
                    f"{node.func.id}() reads the design's wiring, which is "
                    "not known yet: the setting is {value, wired}")
            return wired[node.func.id](*(walk(a) for a in node.args))
        if isinstance(node, ast.Call):
            return _CALLS[node.func.id](*(walk(a) for a in node.args))
        if isinstance(node, ast.Subscript):
            return walk(node.value)[walk(node.slice)]
        raise AssertionError(f"unchecked node {type(node).__name__}")
    return walk(tree)


# -- the loaded hierarchy --------------------------------------------------- #

@dataclass
class Instance:
    """One leaf, at its full path, with everything set on it resolved."""
    path: str
    ref: str
    kind: str                    # "block" | "generator" | "primitive"
    leaf: object                 # the (configured) Block, the callable, the dict
    folder: Path
    build: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    domain: str | None = None
    description: str = ""
    # A vendor primitive's properties, verbatim: its own names, its own
    # value syntax (No_buffer, 74.250), never expressions.
    config: dict = field(default_factory=dict)


@dataclass
class Design:
    """A hierarchy, flattened: leaves by path, edges between leaves."""
    ref: str
    description: str
    parameters: dict             # the top block's own build parameters
    inputs: tuple
    outputs: tuple
    instances: dict              # path -> Instance, in declaration order
    composites: dict             # path -> ref, "" for the top block
    edges: list                  # (source, sink): "path.port" or a top port
    aliases: dict                # "path.register" -> "path.register" it follows
    constraints: list = field(default_factory=list)
    # every constraints file the design is built with, from the blocks that
    # own them, in declaration order: [Path]
    pins: dict = field(default_factory=dict)
    # external pin name -> its declaration ({"dir": ...} or {"interface":
    # true}), gathered from every composite, which is where each is owned
    netlist: dict | None = None
    # the design as a netlist (np2hw.design.netlist): every leaf with its
    # implementation, settings and resolved ports, every bundle and net --
    # the product the checks and the toolchains read
    streams: dict = field(default_factory=dict)
    # top input -> the facts of the stream entering it (bits, max_width,
    # max_height, context), from every input port it passes through on its
    # way in: stated once where it enters, inherited by what wraps it.
    hierarchy: dict = field(default_factory=dict)
    # composite path ("" the top) -> {"ref", "inputs", "outputs", "pins",
    # "children", "connections"}: each composite as it is written, its
    # `when`s resolved -- children by their names in it, connections
    # between pins ((path, port); a composite's own port is (its path,
    # port), an external pin (its owner's path, pin)). One module per
    # composite is written from this; `edges` is the same design flattened.

    @property
    def name(self) -> str:
        """The design's name: its block's, which names its generated top,
        its register map and its pack."""
        return block_name(self.ref)


@dataclass
class _Node:
    path: str
    found: Found
    parent: str | None
    entry: dict                  # the node entry that instantiated it
    children: dict = field(default_factory=dict)   # local name -> path

    @property
    def kind(self):
        return self.found.kind


@dataclass
class _Setting:
    kind: str                    # "build" | "values"
    target: str                  # absolute path of the parameter
    raw: object                  # literal or expression text
    scope: str                   # path of the composite it is written in
    depth: int                   # of that composite: smaller wins
    where: str                   # for messages


def _join(*parts: str) -> str:
    return ".".join(p for p in parts if p)


def tree(ref: str) -> dict:
    """The STRUCTURE under a block: every instance path, composites and
    leaves alike, to what it is -- with no parameter evaluated, so a
    reusable composite can be walked without a design to set it."""
    top = find(ref)
    loader = _Loader()
    loader.expand("", top, None, {}, [])
    return {path: node.found for path, node in loader.nodes.items()}


def load(ref: str, build: dict | None = None, *,
         generate: bool = True) -> Design:
    """Load the block ``ref`` and everything beneath it.

    ``build`` sets the top block's own parameters, as an enclosing composite
    would; a declared default applies where it does not. ``generate=False``
    leaves leaf generators uncalled -- for reading a design's parameters
    before the build has the facts its generators need; such a leaf is
    then wired unchecked, as vendor IP is.
    """
    loader = _Loader(generate=generate)
    return loader.run(ref, build or {})


class _Loader:
    def __init__(self, generate: bool = True):
        self.generate = generate
        self.build_cache: dict[str, object] = {}
        # composites built whole, as one cell of the design around them
        self.opaque: tuple = ()
        # path -> (sub-design, its generated result), built once
        self.elaborations: dict[str, tuple] = {}
        self.elaborating: list[str] = []
        self.whole: dict[str, Instance] = {}
        self.inactive: set[str] = set()
        self.nodes: dict[str, _Node] = {}
        self.settings: dict[tuple[str, str], _Setting] = {}

    # -- expanding the tree ------------------------------------------------- #

    def expand(self, path: str, found: Found, parent, entry, stack):
        if found.ref in stack:
            raise CompositionError(
                f"{path or found.ref}: {found.ref} contains itself "
                f"({' -> '.join([*stack, found.ref])})")
        node = _Node(path, found, parent, entry)
        self.nodes[path] = node
        if found.kind != "composite":
            return
        depth = 0 if not path else path.count(".") + 1
        clash = sorted(set(found.obj.get("parameters", {}))
                       & set(found.obj["nodes"]))
        if clash:
            raise CompositionError(
                f"{found.ref}: {clash} name both a parameter and a node; a "
                "path means one thing")
        for name, child in found.obj["nodes"].items():
            child_path = _join(path, name)
            node.children[name] = child_path
            where = f"{found.ref}: node {name}"
            # Its settings first: what fills a node beneath it may be one of
            # them, and must be known before that node is expanded.
            for kind in ("build", "values"):
                for key, raw in child.get(kind, {}).items():
                    self.add_setting(_Setting(
                        kind, _join(child_path, key), raw, path, depth,
                        f"{where}: {kind} {key}"))
            child_found = self.filling(path, found, child, where)
            if child_found.kind == "generator" and getattr(
                    child_found.obj, "composite", False):
                child_found = _generated(child_found, child, where)
            self.expand(child_path, child_found, path, child,
                        [*stack, found.ref])

    def filling(self, path: str, found: Found, entry: dict,
                where: str) -> Found:
        """The block that fills a node of the composite at `path`: the
        reference its `block` names, or -- where `block` is a bare name --
        the block one of the composite's own build parameters holds, set
        from outside like any other. A parameter's value is an expression;
        `@package.module:block` in it is a reference."""
        name = entry["block"]
        if "." in name:
            return self.found_block(name, where)
        declared = found.obj.get("parameters", {})
        if name not in declared:
            raise CompositionError(
                f"{where}: block {name!r} is neither a reference nor a "
                f"parameter of {found.ref} (parameters {sorted(declared)})")
        target = _join(path, name)
        try:
            ref = self.build_value(target, where, ())
        except CompositionError as error:
            raise CompositionError(
                f"{where}: block {name}: {error} -- a block chosen by a "
                "parameter is chosen from parameters, not from what is "
                "inside a node") from None
        if not isinstance(ref, str):
            raise CompositionError(f"{where}: block {name} = {ref!r}, which "
                                   "names no block; write @package.module")
        chosen = self.found_block(ref, where)
        default = declared[name].get("default")
        if isinstance(default, str):
            first = self.found_block(evaluate(parse(default, where),
                                              lambda n: None), where)
            mine, theirs = _declared_ports(first), _declared_ports(chosen)
            if mine is not None and theirs is not None \
                    and not _same_ports(mine, theirs):
                raise CompositionError(
                    f"{where}: {chosen.ref} cannot fill {target}: its ports "
                    f"are {_describe_ports(theirs)}, and {first.ref}'s, "
                    "which the composite is wired to, are "
                    f"{_describe_ports(mine)}")
        return chosen

    def found_block(self, ref: str, where: str) -> Found:
        try:
            found = find(ref)
        except CompositionError as error:
            raise CompositionError(f"{where}: {error}") from None
        if found.kind == "profile":
            raise CompositionError(f"{where}: {ref} is a profile, not a "
                                   "block")
        return found

    def add_setting(self, setting: _Setting):
        key = (setting.kind, setting.target)
        held = self.settings.get(key)
        if held is None or setting.depth < held.depth:
            self.settings[key] = setting
        elif setting.depth == held.depth:
            raise CompositionError(f"{setting.where}: sets {setting.target} "
                                   f"a second time ({held.where})")

    # -- naming parameters -------------------------------------------------- #

    def split(self, path: str, where: str) -> tuple[_Node, str]:
        """An absolute parameter path, as (the instance it belongs to, the
        rest). Walks instance names as far as they go."""
        node = self.nodes[""]
        parts = path.split(".") if path else []
        i = 0
        while i < len(parts) and parts[i] in node.children:
            node = self.nodes[node.children[parts[i]]]
            i += 1
        rest = ".".join(parts[i:])
        if not rest:
            raise CompositionError(f"{where}: {path} is an instance, not a "
                                   "parameter")
        return node, rest

    def classify(self, path: str, where: str) -> str:
        """"build" or "register", or a refusal naming what exists."""
        node, rest = self.split(path, where)
        kind = node.kind
        if kind == "primitive" and _vendor_ip(node.found.obj):
            # A vendor's IP owns its parameter list, and its tool refuses a
            # name it does not have; restating the list here would be a
            # second copy of the vendor's.
            return "build"
        if kind in ("composite", "primitive"):
            declared = node.found.obj.get("parameters", {})
            if rest in declared:
                return "build"
            children = sorted(node.children)
            raise CompositionError(
                f"{where}: {path}: {node.found.ref} has no parameter "
                f"{rest!r} (parameters {sorted(declared)}"
                + (f", nodes {children})" if children else ")"))
        if kind == "generator":
            accepted = _generator_parameters(node.found.obj)
            if rest in accepted:
                return "build"
            raise CompositionError(
                f"{where}: {path}: {node.found.ref} takes "
                f"{sorted(accepted)}, not {rest!r}")
        block = node.found.obj
        registers = {p.name for p in block.params.params}
        head, _, attr = rest.partition(".")
        if head in registers and not attr:
            return "register"
        if head in registers and attr and "." not in attr:
            return "build"               # a register attribute: gain.frac
        raise CompositionError(
            f"{where}: {path}: block {block.name} has no register {head!r} "
            f"(registers {sorted(registers)})")

    def resolve_name(self, scope: str, name: str) -> str:
        """A path as written in the composite at ``scope``, made absolute.
        An expression reaches only its own composite's parameters and what is
        inside its nodes -- never its surroundings."""
        return _join(scope, name)

    # -- build parameters --------------------------------------------------- #

    def build_value(self, path: str, where: str, active: tuple):
        if path in self.build_cache:
            return self.build_cache[path]
        if path in active:
            cycle = " -> ".join([*active[active.index(path):], path])
            raise CompositionError(f"{where}: a cycle: {cycle}")
        setting = self.settings.get(("build", path))
        if setting is None:
            node, rest = self.split(path, where)
            declared = (node.found.obj.get("parameters", {})
                        if node.kind in ("composite", "primitive") else {})
            if rest in declared and "default" in declared[rest]:
                # A default may be derived: a string is an expression, in
                # the scope of the block that declares it.
                value = self.evaluate_build(
                    declared[rest]["default"], node.path,
                    f"{node.found.ref}: default of {rest}", (*active, path))
            elif node.kind in ("composite", "primitive"):
                raise CompositionError(
                    f"{where}: {path} is never set, and {node.found.ref} "
                    "declares no default for it")
            else:
                raise CompositionError(f"{where}: {path} is never set")
        else:
            value = self.evaluate_build(setting.raw, setting.scope,
                                        setting.where, (*active, path))
        self.build_cache[path] = value
        return value

    def evaluate_build(self, raw, scope: str, where: str, active=()):
        """A literal, or an expression over build parameters in ``scope``,
        as an integer; a list, item by item."""
        if isinstance(raw, list):
            return [self.evaluate_build(item, scope, where, active)
                    for item in raw]
        if not isinstance(raw, str):
            return raw
        tree = parse(raw, where)

        def lookup(name):
            target = self.resolve_name(scope, name)
            fact = self.implementation_fact(target, where)
            if fact is None:
                fact = self.port_fact(target, where)
            if fact is None:
                fact = self.window_fact(target, where)
            if fact is not None:
                return fact
            if self.classify(target, where) != "build":
                raise CompositionError(
                    f"{where}: {raw!r} reads {name}, a register; a build "
                    "parameter changes the hardware and cannot follow a "
                    "value software writes")
            return self.build_value(target, where, active)
        return evaluate(tree, lookup)

    # -- the streams entering ----------------------------------------------- #

    STREAM_FACTS = ("bits", "max_width", "max_height", "context")

    def streams(self, forward: dict, root: str = "") -> dict:
        """For each top input, the facts of the stream entering it,
        gathered from every composite input port it passes through. Each
        fact is stated once -- where the stream enters -- and a composite
        that wraps it inherits it; two ports stating it differently are
        refused, and so is a fact stated on an output, where the stream is
        what the blocks made it."""
        for path, node in self.nodes.items():
            if node.kind != "composite":
                continue
            for port in node.found.obj.get("outputs", []):
                stated = [f for f in self.STREAM_FACTS if f in port]
                if stated:
                    raise CompositionError(
                        f"{node.found.ref}: output {port['name']} states "
                        f"{stated}; an output carries what the blocks "
                        "produce, which is traced, not declared")

        def declared(pin):
            node = self.nodes[pin[0]]
            port = next(p for p in node.found.obj.get("inputs", [])
                        if p["name"] == pin[1])
            where = f"{node.found.ref}: input {pin[1]}"
            facts = {}
            for fact in self.STREAM_FACTS:
                if fact in port:
                    facts[fact] = (list(port[fact]) if fact == "context"
                                   else self.evaluate_build(
                                       port[fact], pin[0], f"{where}: {fact}"))
            return facts, where

        result = {}
        top = self.nodes[root]
        for port in top.found.obj.get("inputs", []):
            facts, source = {}, {}
            pending, seen = [(root, port["name"])], set()
            while pending:
                pin = pending.pop()
                if pin in seen:
                    continue
                seen.add(pin)
                here, where = declared(pin)
                for fact, value in here.items():
                    if fact in facts and facts[fact] != value:
                        raise CompositionError(
                            f"{where}: {fact} = {value}, but "
                            f"{source[fact]} states {facts[fact]} for the "
                            f"same stream (entering {port['name']})")
                    facts[fact], source[fact] = value, where
                for nxt in forward.get(pin, []):
                    if nxt[0] != root and \
                            self.nodes[nxt[0]].kind == "composite":
                        pending.append(nxt)
            result[port["name"]] = facts
        return result

    # -- facts of a node's ports, and composites built whole --------------- #

    FACTS = ("bits", "channels", "data_bits", "max_width", "max_height")

    def port_fact(self, target: str, where: str):
        """``node.port.fact`` -- a fact of the stream at a composite's port,
        or None where `target` names no such thing. On an INPUT it is the
        fact declared where the stream enters; on an OUTPUT it is traced, so
        the composite is built (once) to read it. This is how a composite
        assigns one of its parameters what a node of it produces."""
        node, rest = self.split(target, where)
        port, dot, fact = rest.partition(".")
        if not dot or fact not in self.FACTS or node.kind != "composite":
            return None
        obj = node.found.obj
        inputs = {p["name"] for p in obj.get("inputs", [])}
        outputs = {p["name"] for p in obj.get("outputs", [])}
        if port in inputs:
            _, forward = self._edges(node.path)
            facts = self.streams(forward, node.path)[port]
            if fact in ("bits", "max_width", "max_height"):
                if fact not in facts:
                    raise CompositionError(
                        f"{where}: {target}: the stream entering "
                        f"{node.found.ref}'s {port} states no {fact}")
                return facts[fact]
            channels = int(facts.get("channels", 1))
            return channels if fact == "channels" else \
                int(facts["bits"]) * channels
        if port in outputs:
            if fact in ("max_width", "max_height"):
                raise CompositionError(
                    f"{where}: {target}: geometry is declared where a stream "
                    "enters; an output's is not traced yet")
            _, generated = self.elaborate(node.path, where)
            out = generated.meta["boundary"]["outputs"][port]
            return {"data_bits": out["data_bits"],
                    "channels": out["channels"],
                    "bits": out.get("channel_bits", out["data_bits"])}[fact]
        raise CompositionError(
            f"{where}: {target}: {node.found.ref} has no port {port!r} "
            f"(inputs {sorted(inputs)}, outputs {sorted(outputs)})")

    #: The smallest register window a bus gives a slave: one page, so a host
    #: maps each slave apart from its neighbours.
    PAGE = 0x1000

    def window_fact(self, target: str, where: str):
        """``node.port.window`` -- the address window an AXI4-Lite slave
        port needs: two to the power of its address width, at least a page;
        or None where `target` names no such thing. The width is the
        slave's own statement -- declared on its block, published by its
        generator, or published by its build where it is built whole --
        and reading it builds the slave first, whatever order the design
        lists its nodes in."""
        node, rest = self.split(target, where)
        port, dot, fact = rest.partition(".")
        if not dot or fact != "window":
            return None
        width = None
        if node.kind == "composite":
            ports = self.built_whole(node.path).leaf["ports"]
        else:
            if (self.generate and node.kind == "generator"
                    and getattr(node.found.obj, "leaf", False)):
                node.found = self.generated_leaf(node.path, node)
            obj = node.found.obj if isinstance(node.found.obj, dict) else {}
            ports = obj.get("ports")
            if ports is None:
                for p in obj.get("inputs", []):
                    if p["name"] == port and p.get("bus") == "axilite":
                        width = self.evaluate_build(
                            p.get("width", 32), node.path,
                            f"{where}: {target}")
        if ports is not None:
            width = next((p["width"] for p in ports
                          if p["name"] == f"{port}_awaddr"
                          and p["direction"] == "input"), None)
        if not isinstance(width, int):
            raise CompositionError(
                f"{where}: {target}: {node.found.ref} states no AXI4-Lite "
                f"slave port {port!r} with an address width")
        return max(self.PAGE, 1 << width)

    def subdesign(self, path: str) -> Design:
        """The composite at `path` as a design of its own: the instances
        beneath it, named relative to it, with everything the design around
        it set on them; the edges inside it, its own ports as the boundary;
        the stream entering it, declared where it enters."""
        node = self.nodes[path]
        edges, forward = self._edges(path)
        aliases: dict = {}
        instances = {}
        for p, n in self.nodes.items():
            if (n.kind == "composite" or p in self.inactive
                    or not p.startswith(path + ".")):
                continue
            inst = self.instance(p, n, aliases)
            inst.path = p[len(path) + 1:]
            instances[inst.path] = inst
        obj = node.found.obj
        return Design(
            ref=node.found.ref, description=obj.get("description", ""),
            parameters={name: self.build_value(_join(path, name),
                                               f"{node.found.ref}: {name}",
                                               ())
                        for name in obj.get("parameters", {})},
            inputs=tuple(p["name"] for p in obj.get("inputs", [])),
            outputs=tuple(p["name"] for p in obj.get("outputs", [])),
            instances=instances,
            composites={(p[len(path) + 1:] if p != path else ""):
                        n.found.ref for p, n in self.nodes.items()
                        if n.kind == "composite" and p not in self.inactive
                        and (p == path or p.startswith(path + "."))},
            edges=edges,
            aliases={k[len(path) + 1:]: v[len(path) + 1:]
                     for k, v in aliases.items()},
            streams=self.streams(forward, path))

    def elaborate(self, path: str, where: str):
        """Build the composite at `path` -- model blocks -- once: its
        sub-design and the generated result, the module a design around it
        instantiates. Its own `clock_mhz` parameter, where it has one, is
        the clock its datapath is cut for."""
        if path in self.elaborations:
            return self.elaborations[path]
        if path in self.elaborating:
            raise CompositionError(
                f"{where}: a cycle: building {path} needs "
                f"{' -> '.join([*self.elaborating, path])}")
        self.elaborating.append(path)
        try:
            from np2hw.pipeline import compose
            sub = self.subdesign(path)
            mhz = sub.parameters.get("clock_mhz")
            generated = compose.generate(
                sub, control=True,
                clk_ns=None if mhz is None else 1000.0 / float(mhz))
        finally:
            self.elaborating.pop()
        self.elaborations[path] = (sub, generated)
        return sub, generated

    def opaque_composites(self) -> tuple:
        """The composites built whole: in a design that holds anything but
        model blocks, each largest composite holding model blocks alone.
        A design of model blocks alone is itself what gets built."""
        leaves = [(p, n) for p, n in self.nodes.items()
                  if n.kind != "composite" and p not in self.inactive]
        if all(n.kind == "block" for _, n in leaves):
            return ()

        def models_only(path):
            inside = [n for p, n in leaves if p.startswith(path + ".")]
            return bool(inside) and all(n.kind == "block" for n in inside)
        chosen = []
        for path, node in self.nodes.items():
            if (path and node.kind == "composite"
                    and path not in self.inactive and models_only(path)
                    and not any(path.startswith(c + ".") for c in chosen)):
                chosen.append(path)
        return tuple(chosen)

    def built_whole(self, path: str) -> Instance:
        """The composite at `path` as one leaf: the module its build made,
        with the ports that build published, and its pack beside it."""
        if path in self.whole:
            return self.whole[path]
        from np2hw.pipeline import compose
        from np2hw.design.interfaces import buses
        sub, generated = self.elaborate(path, path)
        files = {f"{sub.name}.v": generated.verilog + "\n",
                 f"{sub.name}_regmap.json":
                     json.dumps(compose.register_map(sub), indent=2) + "\n"}
        ports = generated.meta["ports"]
        buses(ports)                     # its streams are whole
        # constraints the build made for its own module, written beside its
        # Verilog; each says when it applies (np2hw.toolchain reads it)
        from np2hw.design.interfaces import generated_constraints
        crossing = generated_constraints(
            generated.meta.get("constraints", ()), files)
        node = self.nodes[path]
        self.whole[path] = Instance(
            path=path, ref=node.found.ref, kind="primitive",
            leaf={"description": node.found.obj.get("description", ""),
                  "vivado": {"module": generated.top},
                  "ports": ports, "module_parameters": {},
                  "generated": [f"{sub.name}.v"], "files": files,
                  "facts": dict(generated.meta.get("facts", {})),
                  "generated_constraints": crossing,
                  "made_by": {"composite": node.found.ref}},
            folder=node.found.folder,
            domain=node.entry.get("domain"),
            description=node.entry.get("description", ""))
        return self.whole[path]

    # -- the run ------------------------------------------------------------ #

    def run(self, ref: str, top_build: dict) -> Design:
        top = find(ref)
        if top.kind != "composite":
            raise CompositionError(f"{ref} is a {top.kind}, not a composite; "
                                   "a design is a composite of blocks")
        for key, raw in top_build.items():
            self.add_setting(_Setting("build", key, raw, "", -1,
                                      f"load({ref!r}): build {key}"))
        self.expand("", top, None, {}, [])

        # Every build setting names something that exists, and is BUILD.
        for (kind, target), setting in self.settings.items():
            what = self.classify(target, setting.where)
            if kind == "build" and what != "build":
                raise CompositionError(
                    f"{setting.where}: {target} is a register; set it under "
                    "values, not build")
            if kind == "values" and what != "register":
                raise CompositionError(
                    f"{setting.where}: {target} is a build parameter; set "
                    "it under build, not values")

        # What this build has: a node whose `when` is false is not built,
        # and neither is anything inside it.
        self.inactive: set[str] = set()
        for path, node in self.nodes.items():
            when = node.entry.get("when")
            if when is None or any(path.startswith(p + ".")
                                   for p in self.inactive):
                continue
            where = f"{self.nodes[node.parent].found.ref}: node " \
                    f"{path.rpartition('.')[2]}: when"
            if not self.evaluate_build(when, node.parent, where):
                self.inactive.add(path)
        self.inactive = {p for p in self.nodes
                         if any(p == q or p.startswith(q + ".")
                                for q in self.inactive)}

        # Every declared parameter of every composite, evaluated -- an unset
        # one is an error even where nothing reads it yet: it is part of
        # the interface, and the interface is incomplete.
        for path, node in self.nodes.items():
            if path in self.inactive:
                continue
            if node.kind == "composite":
                for name in node.found.obj.get("parameters", {}):
                    self.build_value(_join(path, name),
                                     f"{node.found.ref}: parameter {name}", ())

        # A generator that makes a leaf -- a receiver, a raster -- is called
        # now, with its build settings evaluated, and what it publishes
        # stands in for a declaration: its ports, its module, its Verilog.
        # (A node another one read a fact of is generated already: its
        # kind is no longer "generator".)
        for path, node in self.nodes.items():
            if (self.generate and node.kind == "generator"
                    and path not in self.inactive
                    and getattr(node.found.obj, "leaf", False)):
                node.found = self.generated_leaf(path, node)

        # A block described apart from its implementations is filled by the
        # one its `implementation` parameter chooses: the tool's facts, its
        # own port names, what follows from it.
        for path, node in self.nodes.items():
            if (node.kind == "primitive" and path not in self.inactive
                    and "implementations" in node.found.obj):
                node.found = self.implemented_leaf(path, node)

        # A composite of model blocks inside a design that holds other
        # leaves is built whole -- the ISP inside a board -- and is one
        # cell of it, wired by the ports its build published.
        self.opaque = self.opaque_composites()

        instances, aliases = {}, {}
        for path, node in self.nodes.items():
            if path in self.opaque:
                if self.generate:
                    instances[path] = self.built_whole(path)
                continue
            if (node.kind == "composite" or path in self.inactive
                    or not self.in_scope(path, "", self.opaque)):
                continue
            instances[path] = self.instance(path, node, aliases)

        design = Design(
            ref=top.ref, description=top.obj.get("description", ""),
            parameters={name: self.build_cache[name]
                        for name in top.obj.get("parameters", {})},
            inputs=tuple(p["name"] for p in top.obj.get("inputs", [])),
            outputs=tuple(p["name"] for p in top.obj.get("outputs", [])),
            instances=instances,
            composites={p: n.found.ref for p, n in self.nodes.items()
                        if n.kind == "composite" and p not in self.inactive
                        and p not in self.opaque
                        and self.in_scope(p, "", self.opaque)},
            edges=self.edges(),
            aliases=aliases,
            pins=self.pins(),
            constraints=self.constraints(),
        )
        design.streams = self.streams(self.forward)
        design.hierarchy = {
            path: {"ref": self.nodes[path].found.ref,
                   "inputs": [p["name"] for p in
                              self.nodes[path].found.obj.get("inputs", [])],
                   "outputs": [p["name"] for p in
                               self.nodes[path].found.obj.get("outputs", [])],
                   "pins": [p["name"] for p in
                            self.nodes[path].found.obj.get("pins", [])],
                   "children": {name: child for name, child
                                in self.nodes[path].children.items()
                                if child not in self.inactive},
                   "connections": list(self.local.get(path, []))}
            for path in design.composites}
        if self.generate:
            # the netlist, built once; every wire against the ports of the
            # leaves it joins, read from it, for every toolchain
            from np2hw.design.netlist import build
            from np2hw.design.wiring import check
            design.netlist = build(design)
            check(design)
            self.wire(design)
        return design

    def wire(self, design) -> None:
        """The settings that follow what a node is connected to ({value,
        wired}), evaluated against the netlist: ``same_net('a', 'b', ...)``
        is whether those ports of the node are all on one net."""
        net_of = {pin: i for i, net in enumerate(design.netlist["nets"])
                  for pin in net["pins"]}
        for path, inst in design.instances.items():
            leaf = inst.leaf if isinstance(inst.leaf, dict) else {}
            for key, (raw, where) in leaf.get(
                    "implementation_wired", {}).items():
                def same_net(*ports, path=path, where=where):
                    nets = set()
                    for port in ports:
                        if f"{path}.{port}" not in net_of:
                            raise CompositionError(
                                f"{where}: same_net reads {port}, which is "
                                "on no net here")
                        nets.add(net_of[f"{path}.{port}"])
                    return int(len(nets) == 1)
                if key in inst.config or key in inst.build:
                    raise CompositionError(f"{path}: {key} is set on the "
                                           "node and follows its wiring; a "
                                           "property has one value")
                tree = parse(raw, where)
                inst.config[key] = evaluate(
                    tree, lambda name, path=path, where=where:
                    self.build_value(self.resolve_name(path, name), where,
                                     ()),
                    wired={"same_net": same_net})

    def build_settings(self, path: str, node: _Node) -> dict:
        """The build settings made on one node, evaluated, by their names
        relative to it."""
        return {k[1][len(path) + 1:]: self.build_value(k[1], s.where, ())
                for k, s in self.settings.items()
                if k[0] == "build" and k[1].startswith(path + ".")
                and self.split(k[1], s.where)[0] is node}

    def generated_leaf(self, path: str, node: _Node) -> Found:
        """Call a leaf generator with its node's build settings; its result
        becomes the leaf's declaration (`interfaces.from_result`)."""
        from np2hw.design.interfaces import from_result
        args = self.build_settings(path, node)
        where = f"{path} ({node.found.ref})"
        try:
            result = node.found.obj(**args)
        except (KeyError, ValueError, TypeError) as error:
            raise CompositionError(f"{where}: {error}") from None
        try:
            declaration = from_result(result, node.found.obj)
        except ValueError as error:
            raise CompositionError(f"{where}: {error}") from None
        declaration["made_by"] = {"generator": node.found.ref, "build": args}
        return Found(node.found.ref, "primitive", declaration,
                     node.found.folder)

    def implementation(self, path: str, node: _Node, where: str):
        """(name, implementation) a block's `implementation` parameter
        chooses, or None for a leaf that is its own implementation."""
        obj = node.found.obj
        if isinstance(obj, dict) and "filled_by" in obj:
            return obj["filled_by"]["implementation"], obj["filled_by"]["chosen"]
        if not (isinstance(obj, dict) and "implementations" in obj):
            return None
        name = self.build_value(_join(path, "implementation"), where, ())
        chosen = obj["implementations"].get(name)
        if chosen is None:
            raise CompositionError(
                f"{where}: {path}: {node.found.ref} has no implementation "
                f"{name!r} (implementations {sorted(obj['implementations'])})")
        return name, chosen

    def implementation_fact(self, target: str, where: str):
        """``node.fact`` -- what follows from the implementation filling a
        block (a front end's lane order), or None where `target` names no
        such thing."""
        node, rest = self.split(target, where)
        if node.kind != "primitive" or "." in rest:
            return None
        found = self.implementation(node.path, node, where)
        if found is None or rest not in found[1].get("facts", {}):
            return None
        return found[1]["facts"][rest]

    def implemented_leaf(self, path: str, node: _Node) -> Found:
        """A block filled by its chosen implementation: the block's ports
        and description, wired by the design; the implementation's tool,
        port names and settings, evaluated over the block's parameters."""
        where = f"{path} ({node.found.ref})"
        name, chosen = self.implementation(path, node, where)
        block = node.found.obj
        leaf = {k: v for k, v in block.items()
                if k not in ("implementations", "parameters")}
        for key in ("vivado", "sources", "constraints"):
            if key in chosen:
                leaf[key] = chosen[key]
        leaf["port_map"] = dict(chosen.get("ports", {}))
        leaf["tie"] = dict(chosen.get("tie", {}))
        # the block's parameters at this node, which its port widths read
        leaf["block_parameters"] = {
            name: self.build_value(_join(path, name), where, ())
            for name in block.get("parameters", {})}
        # a port the block has only under some parameters, as this node
        # sets them
        for side in ("inputs", "outputs"):
            leaf[side] = [
                {k: v for k, v in port.items() if k != "when"}
                for port in block.get(side, [])
                if "when" not in port or self.evaluate_build(
                    port["when"], path, f"{where}: {port['name']}: when")]
        config, wired = {}, {}
        for key, raw in chosen.get("config", {}).items():
            here = f"{where}: implementation {name}: config {key}"
            if isinstance(raw, dict):
                if "when" in raw and not self.evaluate_build(
                        raw["when"], path, f"{here}: when"):
                    continue
                if raw.get("wired"):
                    # follows what the node is connected to: evaluated
                    # once the design is wired (Loader.wire)
                    wired[key] = (raw["value"], here)
                    continue
                raw = raw["value"]
            config[key] = self.evaluate_build(raw, path, here)
        leaf["implementation_config"] = config
        leaf["implementation_wired"] = wired
        leaf["filled_by"] = {"implementation": name, "chosen": chosen}
        return Found(node.found.ref, "primitive", leaf, node.found.folder)

    def instance(self, path: str, node: _Node, aliases: dict) -> Instance:
        # A leaf generator left uncalled (generate=False) has no arguments
        # to evaluate: they would be spent making the module, and may read
        # facts only the generated design has.
        build = ({} if node.kind == "generator" and not self.generate
                 else self.build_settings(path, node))
        if isinstance(node.found.obj, dict) and (
                "made_by" in node.found.obj or "filled_by" in node.found.obj):
            # its settings were the generator's arguments, spent making the
            # module, or the block's, spent choosing and setting its
            # implementation; none is a parameter of the module itself
            build = {}
        elif node.kind == "primitive" and isinstance(node.found.obj, dict):
            # A declared default is the parameter's value where nothing sets
            # it -- for every reader of the instance: its port widths, the
            # facts read off it, the module the toolchain instantiates.
            for name, spec in node.found.obj.get("parameters", {}).items():
                if name not in build and "default" in spec:
                    build[name] = self.build_value(
                        _join(path, name), f"{path} ({node.found.ref})", ())
        leaf = node.found.obj
        overrides: dict = {}
        if node.kind == "block":
            for key, value in build.items():
                register, attr = key.split(".")
                overrides.setdefault(register, {})[attr] = value
            try:
                leaf = leaf.configure(overrides or None)
            except (ValueError, KeyError, TypeError) as error:
                raise CompositionError(f"{path}: {error}") from None
        values = {}
        for (kind, target), setting in self.settings.items():
            if kind != "values" or self.split(target, setting.where)[0] \
                    is not node:
                continue
            register = target[len(path) + 1:]
            value = self.register_value(setting, target, aliases)
            if value is not None:
                _check_register(leaf, register, value, setting.where)
                values[register] = value
        if node.kind == "block" and values:
            # A design's values are the register file's RESET values: part
            # of the configured variant, so the map, the RTL's reset and the
            # model's defaults all read them from one place.
            leaf = node.found.obj.configure(overrides or None, resets=values)
        config = dict(node.entry.get("config", {}))
        if isinstance(leaf, dict) and "implementation_config" in leaf:
            if _vendor_ip(leaf):
                config = {**leaf["implementation_config"], **config}
            else:
                # an implementation that is a module: its settings are the
                # module's parameters, as a leaf's build is
                build = dict(leaf["implementation_config"])
        both = sorted(set(config) & set(build))
        if both:
            raise CompositionError(f"{path}: {both} set under both build and "
                                   "config; a property has one value")
        if config and not (node.kind == "primitive"
                           and _vendor_ip(node.found.obj)):
            raise CompositionError(f"{path}: config is a vendor IP's "
                                   "properties, and this is not vendor IP")
        return Instance(path=path, ref=node.found.ref, kind=node.kind,
                        leaf=leaf, folder=node.found.folder, build=build,
                        values=values, domain=node.entry.get("domain"),
                        description=node.entry.get("description", ""),
                        config=config)

    def register_value(self, setting: _Setting, target: str, aliases: dict):
        """A literal, a constant from build parameters, or an alias. None
        for an alias: the register's value is its source's. A list holds
        literals and constants, item by item; an alias is a whole register."""
        if isinstance(setting.raw, list):
            def constant(item):
                if isinstance(item, list):
                    return [constant(i) for i in item]
                if not isinstance(item, str):
                    return item
                tree = parse(item, setting.where)
                for name in references(tree):
                    if self.classify(self.resolve_name(setting.scope, name),
                                     setting.where) == "register":
                        raise CompositionError(
                            f"{setting.where}: {item!r} reads {name}, a "
                            "register; an item of a list is a constant")
                return evaluate(tree, lambda name: self.build_value(
                    self.resolve_name(setting.scope, name), setting.where, ()))
            return constant(setting.raw)
        if not isinstance(setting.raw, str):
            return setting.raw
        tree = parse(setting.raw, setting.where)
        names = references(tree)
        kinds = {name: self.classify(self.resolve_name(setting.scope, name),
                                     setting.where) for name in names}
        registers = [n for n, k in kinds.items() if k == "register"]
        if not registers:
            return evaluate(tree, lambda name: self.build_value(
                self.resolve_name(setting.scope, name), setting.where, ()))
        if len(names) == 1 and _dotted(tree) is not None:
            source = self.resolve_name(setting.scope, names[0])
            if source == target:
                raise CompositionError(f"{setting.where}: {target} follows "
                                       "itself")
            aliases[target] = source
            return None
        raise CompositionError(
            f"{setting.where}: {setting.raw!r} computes with registers "
            f"({', '.join(registers)}). A register may FOLLOW one other "
            "register (an alias); arithmetic between registers is refused "
            "until np2hw can trace it")

    # -- connections -------------------------------------------------------- #

    def ports(self, node: _Node):
        """(inputs, outputs) a node declares, or None where it cannot know
        them before it is generated."""
        if node.kind == "block":
            return set(node.found.obj.ports.inputs), \
                set(node.found.obj.ports.outputs)
        if node.path in self.opaque:
            # built whole: wired by the ports its build published -- and,
            # where nothing is generated, wired unchecked, like a leaf
            # generator that has not been called
            if not self.generate:
                return None
            return _leaf_ports(self.built_whole(node.path).leaf)
        if node.kind == "primitive":
            return _leaf_ports(node.found.obj)
        if node.kind == "composite":
            obj = node.found.obj
            return ({p["name"] for p in obj.get("inputs", [])},
                    {p["name"] for p in obj.get("outputs", [])})
        return None

    def edges(self) -> list[tuple[str, str]]:
        edges, self.forward = self._edges("", self.opaque)
        return edges

    def in_scope(self, path: str, root: str, opaque=()) -> bool:
        """Whether a node is inside the composite at `root` and not hidden
        inside one built whole (`opaque`)."""
        if root and not (path == root or path.startswith(root + ".")):
            return False
        return not any(path.startswith(o + ".") for o in opaque)

    def constraints(self) -> list:
        """Every constraints file of every built block, composite or
        primitive, as ``{"file", "scoped_to"}``: resolved beside the file
        that names it or in the package it names, `scoped_to` the module
        whose every instance it constrains (None: the whole design). An
        entry may also carry `when`, read in that block's own scope."""
        out = []
        for path, node in self.nodes.items():
            if path in self.inactive or not isinstance(node.found.obj, dict):
                continue
            for entry in node.found.obj.get("constraints", []):
                scoped = None
                if isinstance(entry, dict):
                    where = f"{node.found.ref}: constraints {entry['file']}"
                    scope = path if node.kind == "composite" else node.parent
                    if "when" in entry and not self.evaluate_build(
                            entry["when"], scope, where):
                        continue
                    scoped = entry.get("scoped_to")
                    entry = entry["file"]
                f = source_file(node.found.folder, entry).resolve()
                item = {"file": f, "scoped_to": scoped}
                if item not in out:
                    out.append(item)
        return out

    def pins(self) -> dict:
        """Every external pin of the design, from whichever composite owns
        it. Pins are the FPGA's ports, so their names are global: two
        composites declaring one name is refused."""
        found, owner = {}, {}
        for path, node in self.nodes.items():
            if node.kind != "composite" or path in self.inactive:
                continue
            for pin in node.found.obj.get("pins", []):
                name = pin["name"]
                if name in found:
                    raise CompositionError(
                        f"{node.found.ref}: pin {name} is also "
                        f"{owner[name]}'s; the FPGA has one port by a name")
                found[name] = {k: v for k, v in pin.items() if k != "name"}
                if "clock" in pin:
                    # its rate, read in the owning composite's scope
                    clock = dict(pin["clock"])
                    clock["khz"] = int(self.evaluate_build(
                        clock["khz"], path,
                        f"{node.found.ref}: pin {name}: clock"))
                    if clock["khz"] <= 0:
                        raise CompositionError(
                            f"{node.found.ref}: pin {name} states a clock "
                            f"of {clock['khz']} kHz")
                    found[name]["clock"] = clock
                owner[name] = node.found.ref
        return found

    def _edges(self, root: str = "", opaque=()):
        # Every connection inside the composite at `root`, as an edge between
        # PINS: (instance path, port). A composite's boundary port is one pin
        # seen from both sides -- its parent's "child.port" and its own
        # "port" -- so joining edges through composite pins leaves edges
        # between leaves. The composite at `root` is the boundary, its pins
        # named bare; a composite in `opaque` is built whole, so its pins
        # are a leaf's and nothing inside it is joined.
        forward: dict[tuple, list] = {}
        driven: dict[tuple, str] = {}
        # each composite's own connections, as pins, kept beside the
        # flattening: the hierarchy a writer of one module per composite
        # reads (Design.hierarchy)
        self.local: dict[str, list] = {}
        for path, node in self.nodes.items():
            if (node.kind != "composite" or path in self.inactive
                    or path in opaque or not self.in_scope(path, root, opaque)):
                continue
            for entry in node.found.obj.get("connections", []):
                if isinstance(entry, dict):
                    source, sink = entry["from"], entry["to"]
                    when = entry.get("when")
                else:
                    (source, sink), when = entry, None
                where = f"{node.found.ref}: connection {source} -> {sink}"
                if when is not None and not self.evaluate_build(
                        when, path, f"{where}: when"):
                    continue
                a = self.pin(node, source, "source", where)
                b = self.pin(node, sink, "sink", where)
                for end, endpoint in ((a, source), (b, sink)):
                    if end[0] in self.inactive:
                        raise CompositionError(
                            f"{where}: {endpoint} is not built here (its "
                            "node's `when` is false); guard the connection "
                            "with the same `when`")
                if b in driven:
                    raise CompositionError(f"{where}: {sink} is already "
                                           f"driven ({driven[b]})")
                driven[b] = where
                forward.setdefault(a, []).append(b)
                self.local.setdefault(path, []).append((a, b))

        def is_pin(pin):
            return pin[1] in _pin_names(self.nodes[pin[0]])

        def is_leaf(pin):
            return pin[0] == root or pin[0] in opaque \
                or self.nodes[pin[0]].kind != "composite" or is_pin(pin)

        def text(pin):
            if is_pin(pin) and not root:
                return f"pin:{pin[1]}"
            if pin[0] == root:
                return pin[1]
            inside = pin[0][len(root) + 1:] if root else pin[0]
            return f"{inside}.{pin[1]}"

        edges = []

        def follow(start, pin, seen):
            for nxt in forward.get(pin, []):
                if nxt in seen:
                    raise CompositionError(f"{text(start)}: a loop through "
                                           "composite ports")
                if is_leaf(nxt):
                    edges.append((text(start), text(nxt)))
                else:
                    follow(start, nxt, seen | {nxt})

        for pin in list(forward):
            if is_leaf(pin):
                follow(pin, pin, {pin})
        # An open wire at a composite boundary: an output nothing inside
        # drives, or an input something inside reads and nothing outside
        # drives. Either would leave a leaf input silently undriven.
        for path, node in self.nodes.items():
            if (node.kind != "composite" or path in self.inactive
                    or path in opaque or not self.in_scope(path, root, opaque)):
                continue
            for port in node.found.obj.get("outputs", []):
                if not self.port_built(node, port):
                    continue
                if (path, port["name"]) not in driven:
                    raise CompositionError(
                        f"{node.found.ref}: output {port['name']} is "
                        "driven by nothing inside it")
            for port in node.found.obj.get("inputs", []):
                pin = (path, port["name"])
                if path != root and pin in forward and pin not in driven:
                    raise CompositionError(
                        f"{path}: input {port['name']} of "
                        f"{node.found.ref} is used inside it and driven by "
                        "nothing")
        return edges, forward

    def port_built(self, node: _Node, port: dict) -> bool:
        """A port with a `when` exists only where it is true, read in its
        composite's own scope -- the consumer-select field of a control
        word, say, where there is a consumer to select."""
        if "when" not in port:
            return True
        return bool(self.evaluate_build(
            port["when"], node.path,
            f"{node.found.ref}: port {port['name']}: when"))

    def pin(self, node: _Node, endpoint: str, role: str, where: str):
        """An endpoint written in composite ``node``, as a pin. A bare name
        is the composite's own port; ``child.port`` is a node's."""
        # Its own ports as it declares them, even where it is built whole:
        # seen from inside, a composite is its JSON.
        obj = node.found.obj
        inputs = {p["name"] for p in obj.get("inputs", [])}
        outputs = {p["name"] for p in obj.get("outputs", [])}
        if "." not in endpoint and endpoint in _pin_names(node):
            return (node.path, endpoint)      # an external pin: either way
        if "." not in endpoint:
            # The composite's own boundary: from inside, its input is a
            # source and its output a sink.
            own = inputs if role == "source" else outputs
            if endpoint not in own:
                raise CompositionError(
                    f"{where}: {node.found.ref} has no "
                    f"{'input' if role == 'source' else 'output'} "
                    f"{endpoint!r} (it has {sorted(own)})")
            return (node.path, endpoint)
        child, port = endpoint.split(".")
        if child not in node.children:
            raise CompositionError(f"{where}: no node {child!r} (nodes "
                                   f"{sorted(node.children)})")
        target = self.nodes[node.children[child]]
        declared = self.ports(target)
        if declared is not None:
            side = declared[1] if role == "source" else declared[0]
            if port not in side:
                raise CompositionError(
                    f"{where}: {child} ({target.found.ref}) has no "
                    f"{'output' if role == 'source' else 'input'} {port!r} "
                    f"(it has {sorted(side)})")
        return (target.path, port)


def write_generated(design: Design, folder) -> list:
    """Write every file the design's generated leaves made into `folder`,
    and return their paths. Two leaves making one file must make it
    alike -- one module from one set of arguments, shared -- and are
    refused otherwise, since a build would keep whichever came last."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    made, by = {}, {}
    for path, inst in design.instances.items():
        obj = inst.leaf if isinstance(inst.leaf, dict) else {}
        for name, content in obj.get("files", {}).items():
            if name in made and made[name] != content:
                raise CompositionError(
                    f"{path} and {by[name]} both generate {name}, "
                    "differently; give one of them its own module")
            made[name], by[name] = content, path
    written = []
    for name, content in made.items():
        target = folder / name
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)
        written.append(target)
    return written


def _declared_ports(found: Found):
    """A block's port names and kinds, where it states them before it is
    built: ``{"inputs": {...}, "outputs": {...}}``, or None for a generator
    or vendor IP, whose ports exist only once it is made."""
    if found.kind == "block":
        return {"inputs": {n: "stream" for n in found.obj.ports.inputs},
                "outputs": {n: "stream" for n in found.obj.ports.outputs}}
    if found.kind not in ("composite", "primitive"):
        return None
    obj = found.obj
    if found.kind == "primitive" and not ("inputs" in obj or "outputs" in obj):
        return None

    def kind(port):
        if found.kind == "composite":
            return None                  # whatever reaches it
        return "stream" if "stream" in port else port.get("bus", "wire")
    return {side: {p["name"]: kind(p) for p in obj.get(side, [])}
            for side in ("inputs", "outputs")}


def _same_ports(a: dict, b: dict) -> bool:
    """The same names on each side, and the same kind wherever both say."""
    for side in ("inputs", "outputs"):
        if set(a[side]) != set(b[side]):
            return False
        for name, kind in a[side].items():
            other = b[side][name]
            if kind is not None and other is not None and kind != other:
                return False
    return True


def _describe_ports(ports: dict) -> str:
    return "; ".join(f"{side} {sorted(names)}" for side, names in
                     ports.items())


def _leaf_ports(obj: dict):
    """(inputs, outputs) a leaf may be wired by, from its declaration, or
    None where it declares none (vendor IP: its ports are its own).

    A stream or a bus is wired by its name, on its own side. Each of its
    signals, and every plain port, is a pin of a net and may be named on
    either side: a status word TAPS a valid, a select may follow an
    enable. Whether a net then has exactly one driver is the net's
    question, not the endpoint's."""
    if "ports" in obj:
        # a generated leaf: its generator's own port list
        from np2hw.design.interfaces import buses
        pins = {p["name"] for p in obj["ports"]}
        sides = {"inputs": set(), "outputs": set()}
        for name, bus in buses(obj["ports"]).items():
            sides[bus["side"]].add(name)
        return sides["inputs"] | pins, sides["outputs"] | pins
    if not ("inputs" in obj or "outputs" in obj):
        return None
    from np2hw.design.interfaces import expand
    pins, sides = set(), {"inputs": set(), "outputs": set()}
    for side in sides:
        for port in obj.get(side, []):
            signals = expand(port, side, 8)
            pins.update(signals)
            if "stream" in port or "bus" in port:
                sides[side].add(port["name"])
    return sides["inputs"] | pins, sides["outputs"] | pins


def _generated(found: Found, entry: dict, where: str) -> Found:
    """A composite a Python function writes from its node's `config` --
    a GPIO block from its field layout, say: one declaration, and every
    cell, setting and port it implies derived from it. The result is an
    ordinary composite, checked against the schema like one written by
    hand."""
    import jsonschema
    config = entry.get("config", {})
    try:
        block = found.obj(config)
    except (KeyError, ValueError, TypeError) as error:
        raise CompositionError(f"{where}: {found.ref}: {error}") from None
    sub = {"$ref": "#/$defs/composite", "$defs": schema()["$defs"]}
    try:
        jsonschema.validate(instance=block, schema=sub)
    except jsonschema.ValidationError as error:
        at = "/".join(str(p) for p in error.absolute_path) or "(top)"
        raise CompositionError(f"{where}: {found.ref} wrote a composite "
                               f"that is not one: at {at}: {error.message}") \
            from None
    return Found(found.ref, "composite", block, found.folder)


def _shared_subsystems(design: Design) -> dict:
    """Composite instances to emit as ONE module instantiated per instance.

    A composite instantiated more than once, built identically and made of
    blocks alone, is the same sub-graph twice -- a stereo pair's eyes --
    and is emitted once, as the flat format's subsystem was. Addresses are
    allocated per instance either way, so the register map is the same; only
    the emitted structure changes. A composite used once stays spelled out,
    so a design's RTL does not change for wrapping a chain in a block."""
    from np2hw.pipeline.compose import Subsystem
    by_ref: dict = {}
    for path, ref in design.composites.items():
        if path:
            by_ref.setdefault(ref, []).append(path)
    shared = {}
    for ref, paths in by_ref.items():
        if len(paths) < 2:
            continue
        obj = find(ref).obj
        connections = obj.get("connections", [])
        if any(isinstance(c, dict) and "when" in c for c in connections) or \
                any("when" in n for n in obj["nodes"].values()):
            continue
        templates = []
        for path in paths:
            inner = [(i.path[len(path) + 1:], i.leaf)
                     for i in design.instances.values()
                     if i.path.startswith(path + ".")]
            if any("." in name for name, _ in inner) or \
                    len(inner) != len(obj["nodes"]):
                break                       # holds composites, not blocks alone
            templates.append(tuple(inner))
        else:
            if all(t == templates[0] for t in templates):
                edges = tuple((c["from"], c["to"]) if isinstance(c, dict)
                              else tuple(c) for c in connections)
                subsystem = Subsystem(
                    name=block_name(ref),
                    inputs=tuple(p["name"] for p in obj.get("inputs", [])),
                    outputs=tuple(p["name"] for p in obj.get("outputs", [])),
                    nodes=templates[0], edges=edges)
                shared.update({path: subsystem for path in paths})
    return shared


def _vendor_ip(obj) -> bool:
    """A leaf whose settings pass straight to its tool -- vendor IP, or a
    procedure -- which checks them: restating its list here would be a
    second copy of the tool's."""
    v = obj.get("vivado", {}) if isinstance(obj, dict) else {}
    return "ip" in v or "tcl" in v


def _pin_names(node) -> set:
    obj = node.found.obj
    if node.kind != "composite":
        return set()
    return {p["name"] for p in obj.get("pins", [])}


def _generator_parameters(fn) -> set[str]:
    return {name for name, p in inspect.signature(fn).parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def _check_register(block, register: str, value, where: str) -> None:
    """A value must be the register's shape and fit its width."""
    param = next(p for p in block.params.params if p.name == register)
    low, high = param.limits

    def shape_of(v):
        if isinstance(v, list):
            inner = {shape_of(x) for x in v}
            if len(inner) != 1:
                raise CompositionError(f"{where}: {register}: a ragged array")
            return (len(v), *inner.pop())
        return ()

    got = shape_of(value)
    if got != tuple(param.shape):
        raise CompositionError(f"{where}: {register} has shape "
                               f"{tuple(param.shape)}, given {got}")

    def check(v):
        if isinstance(v, list):
            for x in v:
                check(x)
        elif not low <= v <= high:
            raise CompositionError(
                f"{where}: {v} does not fit {register} ({param.q_format}, "
                f"{low}..{high})")
    check(value)


def assemble(name: str, blocks: dict, edges, *, inputs=("in",),
             outputs=("out",), stream: dict) -> Design:
    """A design built in code: ``blocks`` ``{path: Block}`` wired by
    ``edges`` ``[(source, sink), ...]``, every input entered by ``stream``
    (``bits``, ``max_width``, ``max_height``, and optionally ``channels``
    and ``context``).

    For what a library file cannot name -- a chain cut out of a loaded
    design to run as a twin, a block a test declares on the spot. It is the
    same Design the loader makes, built by the same steps afterwards; a
    design anyone else will build is a composite in JSON."""
    return Design(
        ref=name, description="", parameters={},
        inputs=tuple(inputs), outputs=tuple(outputs),
        instances={path: Instance(path=path, ref=path, kind="block",
                                  leaf=block, folder=Path("."))
                   for path, block in blocks.items()},
        composites={"": name}, edges=[tuple(e) for e in edges], aliases={},
        streams={port: dict(stream) for port in inputs})


def source_file(folder: Path, entry: str) -> Path:
    """Where a leaf's source file is: beside its library file, or -- written
    ``package:file`` -- shipped in an importable package, as np2hw ships
    its fixed modules (``np2hw.hdl:np2hw_skid.v``)."""
    package, colon, name = entry.partition(":")
    if colon and "/" not in package:
        from importlib.resources import files
        try:
            found = files(package).joinpath(name)
        except ModuleNotFoundError as error:
            raise CompositionError(f"source {entry}: {error}") from None
        if not found.is_file():
            raise CompositionError(f"source {entry}: {package} has no {name}")
        return Path(str(found))
    return Path(folder) / entry

