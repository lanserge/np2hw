# IP-XACT: the interfaces, in a form a tool can read

A generated module's ports are wires. Which of them belong together, and
what protocol they speak, is a fact the module knows and everything
downstream has to work out — a block-design writer by matching names, a
human by reading documentation, an SoC assembly tool not at all.

np2hw states it instead. Every AXI4-Lite port names its bundle, every
clock names the bus it drives, every reset its polarity, and every
stream port names np2hw's own bus definition. The rest of this document
is what that buys and how to use it.

## The two standards in play

**IP-XACT** (IEEE 1685) describes a component: its files, ports, bus
interfaces, memory maps and parameters, as XML. It is what SoC assembly
tools read to connect IP automatically instead of by hand.

**SystemRDL** describes registers, and np2hw emits that too
([`systemrdl()`](api.md)). The two overlap on memory maps; SystemRDL is
the better description of a register file and converts to IP-XACT when a
flow wants one document.

Nothing here duplicates either. The component is an *aggregation* of
facts already published elsewhere — the `.core` manifest names the
files, the traced interface names the ports, the register map names the
addresses — so no fact is stated twice.

## Getting a component: you already have one

The port attributes are what an IP packager reads, so a packaged
component falls out of the RTL with no extra work:

```tcl
create_project probe /tmp/probe -part <part>
set_property ip_repo_paths [list <busdef_dir>] [current_project]
update_ip_catalog -rebuild
add_files <generated>.v
set_property top <toplevel> [current_fileset]
ipx::package_project -root_dir <out> -vendor <you> -library <lib> \
    -taxonomy /UserIP -import_files -force
```

`component.xml` is IP-XACT. On a generated ISP that yields one AXI4-Lite
slave with all 17 ports mapped, a memory map whose range is derived from
the RTL, the bus clock and reset as their own typed interfaces, and each
stream as one interface with its direction inferred.

## The stream's own bus definition

AXI4-Lite is understood everywhere because a standard definition
describes it. The elastic stream is np2hw's own, so without a definition
it arrives as six loose wires per stream and the grouping is guesswork.

`np2hw.ipxact` renders the pair IP-XACT asks for:

- the **bus definition** — that a protocol exists, point to point
  (`maxMasters` 1, `maxSlaves` 1: a beat has one producer and one
  consumer; fan-out is `tee`, a module, so each copy keeps its own
  backpressure) and not addressable (a stream carries no address —
  position rides SOF/EOL/LAST);
- the **abstraction definition** — how *this* encoding names its wires,
  which direction each takes on master and slave, and which are
  required.

```python
from np2hw import ipxact
ipxact.write_definitions("busdef/")   # two .xml files
```

Point a packager's IP repository at that directory before packaging.

**The flags are `optional`, and that is load-bearing.** A core takes SOF
to re-anchor and regenerates the rest, so an input stream carries fewer
flags than the output it feeds. Declaring them required would force
every conforming module to carry all six wires; optional lets a tool
connect the three that are always there and match whichever flags are
present.

**The master is the source.** Everything travels with the data except
READY, which is the sink answering — the handshake's whole asymmetry,
and what lets a packager tell a consumer from a producer without being
told.

## One declaration behind both

`np2hw.stream.STREAM_SIGNALS` is the protocol: the signals, their
widths, which the source drives, and whether each is required. The XML
is rendered from it and the Verilog attributes are emitted from it, so a
tool's view of the protocol and the RTL's cannot drift apart.

That mattered before it was written down. `gen/netlist.py` in a
downstream project groups stream ports by matching
`(.+)_(valid|ready|data|sof|eol|last)$` and deciding a prefix is a
stream if it carries valid *and* ready — inference, from names, in a
project's own tooling. A declaration is what replaces a guess.

## Which modules declare

Everything a design instantiates **by hand**: the composed top from
`control_wrap`, and the plumbing — `skid` (`s`/`m`), `tee` (`in`/`a`/`b`),
`cdc_fifo` (`in`/`out`, each clock naming the face it carries) and
`bayerlink_in` (`out`).

A generated core's internal blocks do not, and that is deliberate. They
are wired by `compose()`, which emitted both ends of every connection —
nothing there is inferring anything, so a declaration would add noise
without removing a guess. The rule is that a bundle is declared wherever
something that did **not** generate the RTL has to work the grouping out.

`tee`'s `en_a`/`en_b`/`torn_a`/`torn_b` stay loose for the same reason
in reverse: they are the tee's own controls, not the stream protocol.

## Version

Written against **IEEE 1685-2009** (the SPIRIT namespace), which is what
the FPGA vendor packagers read and write. 1685-2014 and 1685-2022 exist
and differ mainly in namespace and in renamed elements; a consumer
requiring one of those wants a converted copy, and the conversion is
mechanical because nothing emitted here is vendor-specific.
