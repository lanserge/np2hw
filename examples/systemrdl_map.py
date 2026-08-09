"""A hierarchical register map, rendered as SystemRDL and elaborated back.

The flat `Reg` list is what a register-file emitter needs; it is not what a
design IS. A design has block TYPES and INSTANCES of them at bases, and once
that hierarchy is flattened, everything downstream that wanted it -- a UVM
register model, IP-XACT, generated docs -- has to reconstruct it by guessing at
name prefixes. So the hierarchy is kept:

    RegBlock     "gain_block": these registers, at these LOCAL offsets
    RegInstance  "left.g" is a gain_block at 0x100
    AddrMap      the whole design; flatten() only where an emitter needs it

and `systemrdl()` renders it in the Accellera format an integration team feeds
their own tooling (the PeakRDL exporters produce UVM/IP-XACT/C headers from it
-- separate tools the user runs, not dependencies of np2hw).

Three claims, each checked rather than asserted:

  ONE DEFINITION   two instances of one type produce ONE regfile component,
                   instantiated twice -- the same statement compose() makes
                   with modules.
  SAME ADDRESSES   axil_regfile(addrmap) decodes each register at exactly the
                   address the AddrMap placed it -- one structure, two
                   renderings, no drift.
  ROUND TRIP       if systemrdl-compiler (the reference implementation) is
                   installed, the .rdl elaborates and hands back the same
                   addresses, plus the caller's opaque `properties` as typed
                   user-defined properties.

Run:  python examples/systemrdl_map.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _harness import BUILD
from np2hw import AddrMap, Reg, RegBlock, RegInstance, axil_regfile, systemrdl

# One TYPE...
gain_block = RegBlock(
    "gain_block",
    regs=(
        Reg("id_version", 32, offset=0x0, access="ro", value=0xAB010100,
            description="block ID and version; read before writing anything"),
        Reg("gain", 8, offset=0x4, reset=16,
            description="Q4.4 gain; 16 is unity",
            properties={"q_format": "Q4.4", "frac": 4}),
        Reg("offset", 12, offset=0x8, signed=True,
            description="signed offset applied after the gain",
            properties={"q_format": "Q12.0", "frac": 0}),
    ),
    size=0x100,
    description="one gain stage",
)

# ...two INSTANCES.
addrmap = AddrMap(
    "stereo_ctrl",
    instances=(
        RegInstance("left.g", gain_block, base=0x100),
        RegInstance("right.g", gain_block, base=0x200),
    ),
)


def main():
    text = systemrdl(addrmap)
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, "stereo_ctrl.rdl"), "w") as fh:
        fh.write(text)

    checks = []

    def check(label, ok, detail=""):
        checks.append(ok)
        print(f"  {label:<52} {'PASS' if ok else 'FAIL'}"
              + (f"  {detail}" if detail else ""))

    print("systemrdl_map:")
    check("one regfile definition for two instances",
          text.count("regfile gain_block_rf {") == 1
          and text.count("gain_block_rf left_g @ 0x0100;") == 1
          and text.count("gain_block_rf right_g @ 0x0200;") == 1)
    check("read-only identity is a constant (sw=r; hw=na)",
          "sw = r; hw = na;" in text and "reset = 0xAB010100;" in text)
    check("caller properties rendered as declared UDPs",
          "property q_format { type = string; component = field; };" in text
          and 'q_format = "Q4.4";' in text)

    # The regfile emitter and the AddrMap are two renderings of one structure.
    expected = {r.name: r.offset for r in addrmap.flatten()}
    rf = axil_regfile(addrmap, addr_bits=addrmap.addr_bits())
    check("axil_regfile decodes at the AddrMap's addresses",
          rf["offsets"] == expected, f"{len(expected)} registers")

    # Round trip through the reference compiler, where available.
    try:
        from systemrdl import RDLCompiler
        from systemrdl.node import FieldNode, RegNode
    except ImportError:
        print("  (systemrdl-compiler not installed; textual checks only -- "
              "`pip install systemrdl-compiler` for the elaboration check)")
    else:
        compiler = RDLCompiler()
        compiler.compile_file(os.path.join(BUILD, "stereo_ctrl.rdl"))
        root = compiler.elaborate()
        addresses, formats = {}, {}
        for node in root.descendants(unroll=True):
            if isinstance(node, RegNode):
                path = node.get_path().split(".", 1)[1].replace(".", "_")
                addresses[path] = node.absolute_address
            elif isinstance(node, FieldNode):
                q = node.get_property("q_format", default=None)
                if q is not None:
                    formats[node.get_path()] = q
        check("elaboration returns the AddrMap's addresses",
              addresses == expected)
        check("q_format comes back TYPED from elaboration",
              "Q4.4" in formats.values())

    ok = all(checks)
    print("\n" + ("SYSTEMRDL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
