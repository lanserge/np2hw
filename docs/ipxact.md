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

## How a tool binds the XML to a wire

Nothing is matched by name. The binding is stated on the port itself,
and it is three space-separated fields inside one attribute:

```verilog
, (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 sensor VALID" *)
  input  wire        sensor_valid
```

The fields are the **VLNV** of the abstraction definition, the **name of
the interface instance** this port joins, and the **logical port** it
plays within it. Each answers a different question.

**1. The VLNV says which document.** A packager resolves it against its
IP catalogue, which is why the catalogue has to be pointed at the
definitions before the RTL is added. Note the `_rtl` suffix: this names
the *abstraction* definition (`np2hw_stream_rtl`), not the bus
definition (`np2hw_stream`). The abstraction reaches the bus through its
own `busType` element, so a port names the encoding and the encoding
names the protocol.

**2. The bundle name says which interface.** Every port carrying
`sensor` becomes one interface instance called `sensor`, by string
equality alone — no prefix is parsed out of the port names. That is what
makes the label free to differ from the ports: it can be renamed to
avoid a reserved word while the ports, which other designs wire by name,
stay exactly as they were.

**3. The logical name says which signal of the protocol.** `VALID` is
looked up as a `logicalName` in the abstraction definition. **The
physical name is never read.** `sensor_valid`, `foo_v` and `q` bind
identically; only the attribute decides.

### Master and slave are solved, not declared

The abstraction definition gives every logical port a direction under
each role, so the role follows from the directions the RTL already has:

| port | dir | logical | onMaster | onSlave | fits |
| --- | --- | --- | --- | --- | --- |
| `sensor_valid` | input | VALID | out | in | slave |
| `sensor_ready` | output | READY | in | out | slave |
| `sensor_data` | input | DATA | out | in | slave |

All of them agree, so this end is a sink. Reverse every direction and
the same XML yields a master. A role that is inferred this way cannot
contradict the RTL, because it is computed from it — which is the
argument for declaring directions per role rather than tagging a module
"master" somewhere and hoping the ports match.

`presence` finishes the job: a missing `required` signal is an error, a
missing `optional` one is simply a stream that does not carry that flag.

**Inference does still exist — for ports with no attribute.** That is
the failure mode these declarations displace, and publishing a
definition briefly makes it worse rather than better. The next section
is what that cost.

## Consuming it: what a real block design taught

Authoring the definition is the easy half. Handing it to a tool taught
three things, each of which cost a run to find and none of which is
guessable from the XML.

**The definition has to be in the consumer's catalogue.** A port naming
a VLNV the project has never been given is not ignored quietly: Vivado
reports every one of them as a CRITICAL WARNING (`IP_Flow 19-4976`,
"Interface VLNV not found ... Ignoring X_INTERFACE_INFO parameter") and
falls back to loose pins. Nine ports, nine warnings, on a board whose
build had none of that class. So a project that consumes these modules
must point its IP repository at the directory `write_definitions()`
wrote, before adding the RTL.

**An interface may not be named after a reserved word.** `in` and `out`
are the obvious names for a stream's two ends and both are HDL keywords;
an interface called either is `BD 41-1367`. So the INTERFACE is renamed
-- `in_stream`, `out_stream` -- and the PORTS are not, because ports are
the module's and renaming them would ripple through every design that
wires them. Whoever reads the attributes back should group by the port
name, not by the label: they are free to differ, and only the label has
to be legal. `bundle_name()` is where that rule lives.

**Publishing a definition hands the tool the means to guess.** Once a
stream bus definition is in the catalogue, a tool will happily infer one
from port names alone -- and then report the required signals the
"interface" is missing. A probe with `rx_valid` and no handshake became
an inferred stream missing its READY (`IP_Flow 19-1965`). A port that is
not part of an interface now says so, with `X_INTERFACE_IGNORE`. The
irony is worth stating plainly: the declarations exist to stop a
grouping being guessed from names, and publishing them briefly enabled
exactly that guess one layer up.

### What it is worth, measured

On a composed ISP, packaged with the definition in the catalogue:

| | before | after |
| --- | --- | --- |
| ports claimed by no interface | 16 | **4** |
| stream interfaces | 0 | **2** (`isp_in` slave, `isp_out` master) |

The four that remain are stream context -- not a bus, and correctly
unclaimed. Master and slave were **inferred from the port directions**
rather than declared, which is what the abstraction definition's
`onMaster`/`onSlave` sections are for.

In the block design that instantiates these modules directly, eighteen
stream interfaces are inferred and used, and the build's critical
warnings are unchanged from before any of this existed.

## Version

Written against **IEEE 1685-2009** (the SPIRIT namespace), which is what
the FPGA vendor packagers read and write. 1685-2014 and 1685-2022 exist
and differ mainly in namespace and in renamed elements; a consumer
requiring one of those wants a converted copy, and the conversion is
mechanical because nothing emitted here is vendor-specific.
