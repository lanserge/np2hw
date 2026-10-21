# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""IP-XACT for the elastic stream: the protocol, said in a tool's language.

An AXI4-Lite port is understood by every integration tool because a
standard bus definition describes it, and np2hw's register files now name
that definition on every port. The stream has no such standard: it is
np2hw's own, so an integrator handed a generated core sees six loose
wires per stream and has to read documentation to know they belong
together -- and a block-design writer has to GUESS the grouping from port
names, which is what `gen/netlist.py` does with a regex.

A bus definition is how that stops being a guess. IP-XACT splits it in
two, and the split is the useful part:

  * the BUS DEFINITION says a protocol exists, whether it is addressable,
    and how many masters and slaves a connection may have -- nothing about
    signals, because a protocol outlives any one encoding of it;
  * the ABSTRACTION DEFINITION says how THIS encoding names its wires,
    which are required, which are optional, and which direction each takes
    on the master and on the slave.

Both are rendered from :data:`np2hw.stream.STREAM_SIGNALS`, which the
Verilog emitters also read to attach the matching port attributes. One
declaration, two renderings, so a tool's view of the protocol and the
RTL's cannot drift apart.

## What would change, and what would not

Two things are foreseen and neither wants anticipating in the shape of
this module today.

A **component** -- the document describing the ISP itself: its files,
ports, parameters and memory map -- would be one more `build` function
handed to :func:`_document`, because every IP-XACT document shares the
same skeleton and that skeleton is already factored out. It would own no
new fact: the traced boundary, the register map and the file list are
published elsewhere already, and a component aggregates them. A vendor
packager produces one today from the RTL, which is why this has not been
written; that output is tied to that vendor's bus definitions and is no
use to a flow that cannot resolve them.

An **AXI4-Lite bus definition of our own** would be a second call to
:func:`bus_definition`'s shape. There is no universally published
neutral one -- each vendor ships its own -- so a neutral component
needs it.

A **newer schema** (1685-2014, 1685-2022) is NOT `NS` as a parameter,
and the constant is deliberate. The revisions rename and drop elements;
a module that let a caller swap the namespace would look like it handled
versions while emitting 2009 structure under a 2022 name, which is worse
than not offering it. It wants a second renderer sharing this
declaration.

And the definition functions take no arguments ON PURPOSE. If a caller
could pass its own signal list, it could describe a protocol the Verilog
emitters do not attach -- which is the drift this module exists to make
impossible. The absence of a parameter is the guarantee.

Written against **IEEE 1685-2009** (the SPIRIT namespace), which is what
the FPGA vendor packagers read and write today. 1685-2014 and 1685-2022
exist and differ mainly in namespace and in element names dropped or
renamed; a consumer that requires one of those wants a converted copy,
and the conversion is mechanical because nothing here is vendor-specific.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .stream import (STREAM_VLNV, STREAM_RTL_VLNV, STREAM_SIGNALS, bundle_name,
                     stream_port_attribute)

__all__ = ["NS", "abstraction_definition", "bundle_name",
           "bus_definition", "stream_port_attribute", "vlnv_string",
           "write_definitions", "STREAM_VLNV", "STREAM_RTL_VLNV"], STREAM_VLNV

NS = "http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"

# ElementTree's prefix registry is PROCESS-WIDE, so this reaches outside
# the module. Done once here rather than inside each build, where it
# would be a side effect in a function that otherwise reads as pure --
# and where a reader would have to notice it was happening at all.
ET.register_namespace("spirit", NS)


def _tag(name: str) -> str:
    """A namespaced element name, in ElementTree's notation."""
    return f"{{{NS}}}{name}"


def _vlnv_elements(parent, vlnv) -> None:
    """The four elements that identify anything in IP-XACT."""
    for name, value in zip(("vendor", "library", "name", "version"), vlnv):
        ET.SubElement(parent, _tag(name)).text = value


def _document(root_name: str, vlnv, build) -> str:
    """Serialise one IP-XACT document.

    Built as a tree rather than as text, which is not tidiness: an
    element's text is ESCAPED on the way out. These documents carry
    prose -- what each signal means -- and a description containing an
    ampersand or a less-than is a reasonable thing to write and would
    silently produce a file no parser accepts. Correct by construction
    beats correct by nobody having typed the wrong character yet.
    """
    root = ET.Element(_tag(root_name))
    _vlnv_elements(root, vlnv)
    build(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def vlnv_string(vlnv) -> str:
    """`vendor:library:name:version`, the form an attribute names it by."""
    return ":".join(vlnv)


def bus_definition() -> str:
    """The protocol's existence: point to point, not addressable.

    `directConnection` true and one master to one slave, because a stream
    is a pipe: a beat has exactly one producer and one consumer. Fanning
    one output to several sinks is `tee`'s job, and it is a MODULE rather
    than a bus property precisely because the copies then have their own
    backpressure -- a bus definition claiming many slaves would say the
    protocol arbitrates, and it does not.

    `isAddressable` false: a stream carries no address. Position is the
    frame's, and the frame's structure rides SOF/EOL/LAST.
    """
    def build(root):
        for name, value in (("directConnection", "true"),
                            ("isAddressable", "false"),
                            ("maxMasters", "1"), ("maxSlaves", "1")):
            ET.SubElement(root, _tag(name)).text = value
        ET.SubElement(root, _tag("description")).text = (
            "np2hw elastic stream: one sample per beat, valid/ready "
            "handshake, with framing flags riding the same handshake. A "
            "beat transfers on the cycle where VALID and READY are both "
            "high; VALID may not be withdrawn before it. Point to point "
            "-- fan-out is a tee module, so each copy keeps its own "
            "backpressure.")
    return _document("busDefinition", STREAM_VLNV, build)


def abstraction_definition() -> str:
    """How this encoding names its wires, and which are optional.

    The flags are `optional` and that is a real distinction rather than
    politeness: a core takes SOF to re-anchor and regenerates the rest,
    so an input stream carries fewer flags than the output it feeds.
    Declaring them required would make every conforming module carry all
    six wires; declaring them optional lets a tool connect the three that
    are always there and match the flags that happen to be present.
    """
    def build(root):
        bus = ET.SubElement(root, _tag("busType"))
        for name, value in zip(("vendor", "library", "name", "version"),
                               STREAM_VLNV):
            bus.set(_tag(name), value)
        ports = ET.SubElement(root, _tag("ports"))
        for suffix, logical, width, from_source, presence, doc in STREAM_SIGNALS:
            port = ET.SubElement(ports, _tag("port"))
            ET.SubElement(port, _tag("logicalName")).text = logical
            ET.SubElement(port, _tag("description")).text = doc
            wire = ET.SubElement(port, _tag("wire"))
            if suffix == "data":
                # a qualifier tells a tool what a wire IS, beyond its name
                qual = ET.SubElement(wire, _tag("qualifier"))
                ET.SubElement(qual, _tag("isData")).text = "true"
            # The master is the SOURCE of the stream. Everything but
            # READY travels with the data; READY is the sink answering.
            for side, direction in (("onMaster", "out" if from_source else "in"),
                                    ("onSlave", "in" if from_source else "out")):
                el = ET.SubElement(wire, _tag(side))
                ET.SubElement(el, _tag("presence")).text = presence
                if isinstance(width, int):
                    # DATA and CTX are each port's own width, stated there
                    ET.SubElement(el, _tag("width")).text = str(width)
                ET.SubElement(el, _tag("direction")).text = direction
    return _document("abstractionDefinition", STREAM_RTL_VLNV, build)


def write_definitions(directory) -> dict:
    """Write both files, named as the tools expect to find them.

    Point a packager's IP repository at the directory and a stream port
    stops being six loose wires.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {
        "bus": directory / f"{STREAM_VLNV[2]}.xml",
        "abstraction": directory / f"{STREAM_RTL_VLNV[2]}.xml",
    }
    written["bus"].write_text(bus_definition())
    written["abstraction"].write_text(abstraction_definition())
    return written
