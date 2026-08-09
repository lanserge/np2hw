"""The netlist as a validated structure, before anything is emitted.

compose() wires instances and resolves the handshake at EMISSION time. This
module is the same knowledge one stage earlier: a graph of named nodes with
stream ports, checked for the failures a streaming design can actually have,
while it is still cheap to reject. The rules live here because they are rules
about np2hw's handshake -- an application that re-derives "when may an output
fan out" keeps a copy of the emitter's law, and a copy is free to disagree with
it. That is not hypothetical: the fork rule was implemented twice before this
module existed.

Each validation is a hardware failure, not a tidiness rule:

  UNDRIVEN INPUT    a stream that never arrives; the node waits forever.
  TWO DRIVERS       a short: two sources resolving onto one wire.
  CYCLE             cannot be scheduled in a feed-forward pipeline.
  UNBUFFERED FORK   an output may feed ONE consumer that applies backpressure,
                    plus any number of SINKS (nodes with no outputs, which
                    never stall). Two blocking consumers need an element that
                    buffers one side while the other stalls; there is none yet,
                    so the fork is refused rather than emitted as something
                    that deadlocks the first time a branch stalls -- a bug that
                    only appears under load.

What a node MEANS -- which block, what its streams carry, where its registers
sit -- is the application's. This module knows only names, ports and edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Endpoint:
    """One end of an edge: a node port, or an external (top-level) stream."""

    node: str | None                # None for an external input or output
    port: str

    def __str__(self) -> str:
        return self.port if self.node is None else f"{self.node}.{self.port}"


@dataclass(frozen=True)
class Node:
    """A named vertex with stream ports. What it does is not this layer's business."""

    name: str
    inputs: tuple[str, ...] = ("in",)
    outputs: tuple[str, ...] = ("out",)

    @property
    def is_sink(self) -> bool:
        """No outputs: it observes a stream and must never stall it."""
        return not self.outputs

    def has(self, port: str) -> bool:
        return port in self.inputs or port in self.outputs


@dataclass
class Netlist:
    """Nodes, edges and external streams, with the streaming-contract checks.

    ``name`` appears in every error, because a message that cannot say WHICH
    design is broken makes the reader go find out.
    """

    name: str
    external_inputs: tuple[str, ...] = ()
    external_outputs: tuple[str, ...] = ()
    _nodes: dict = field(default_factory=dict)
    _edges: list = field(default_factory=list)

    # -- construction --------------------------------------------------------- #

    def add(self, node: Node) -> Node:
        if node.name in self._nodes:
            raise ValueError(
                f"{self.name}: node {node.name!r} is already present; a second "
                "instance needs its own name")
        self._nodes[node.name] = node
        return node

    def connect(self, source: str, sink: str) -> None:
        """Add an edge. Endpoints are ``node.port`` or an external stream name.

        Node names may themselves be dotted ("left.blacklevel"), so the port is
        whatever follows the LAST dot and the rest must name a node.
        """
        self._edges.append((self._endpoint(source, "source"),
                            self._endpoint(sink, "sink")))

    def _endpoint(self, text: str, role: str) -> Endpoint:
        if text in self.external_inputs or text in self.external_outputs:
            return Endpoint(None, text)
        node, _, port = text.rpartition(".")
        if not node:
            raise KeyError(
                f"{role} {text!r} is neither an external stream "
                f"({list(self.external_inputs) + list(self.external_outputs)}) "
                "nor a `node.port` reference")
        if node not in self._nodes:
            raise KeyError(
                f"{role} {text!r}: no node {node!r}; this netlist has "
                f"{sorted(self._nodes)}")
        found = self._nodes[node]
        if not found.has(port):
            raise KeyError(
                f"{role} {text!r}: node {node!r} has no port {port!r}; its "
                f"inputs are {list(found.inputs)} and its outputs are "
                f"{list(found.outputs)}")
        return Endpoint(node, port)

    # -- queries --------------------------------------------------------------- #

    @property
    def nodes(self) -> list[Node]:
        """In insertion order, which the caller may treat as declaration order."""
        return list(self._nodes.values())

    @property
    def edges(self) -> list[tuple[Endpoint, Endpoint]]:
        return list(self._edges)

    def node(self, name: str) -> Node:
        try:
            return self._nodes[name]
        except KeyError:
            raise KeyError(f"no node {name!r}; this netlist has "
                           f"{sorted(self._nodes)}") from None

    def consumers(self, source: Endpoint) -> list[Endpoint]:
        return [sink for src, sink in self._edges if src == source]

    def driver(self, sink: Endpoint):
        for src, dst in self._edges:
            if dst == sink:
                return src
        return None

    def is_tap(self, sink: Endpoint) -> bool:
        """Whether this consumer absorbs the stream without ever stalling it.

        An external output carries a real `ready` back into the design, so it
        is never a tap; a SINK node accepts a beat every cycle by contract.
        """
        if sink.node is None:
            return False
        return self._nodes[sink.node].is_sink

    # -- validation ------------------------------------------------------------ #

    def validate(self) -> None:
        """Every check the emitter's handshake semantics imply. Raises on the first."""
        driven: dict[str, list[str]] = {}
        for source, sink in self._edges:
            driven.setdefault(str(sink), []).append(str(source))

        for name, sources in driven.items():
            if len(sources) > 1:
                raise ValueError(
                    f"{self.name}: {name} is driven by {sources} -- an input "
                    "may have exactly one driver")

        for node in self.nodes:
            for port in node.inputs:
                if str(Endpoint(node.name, port)) not in driven:
                    raise ValueError(
                        f"{self.name}: {node.name}.{port} is not connected; "
                        "every input needs a driver")
        for name in self.external_outputs:
            if name not in driven:
                raise ValueError(
                    f"{self.name}: external output {name!r} is not connected")

        for node in self.nodes:
            for port in node.outputs:
                source = Endpoint(node.name, port)
                blocking = [sink for sink in self.consumers(source)
                            if not self.is_tap(sink)]
                if len(blocking) > 1:
                    raise ValueError(
                        f"{self.name}: {source} drives "
                        f"{[str(s) for s in blocking]}, which both apply "
                        "backpressure. A fork between two real datapaths needs "
                        "a buffering fork element, which does not exist yet; a "
                        "tap to a sink node (one with no outputs) is free and "
                        "any number are allowed.")

        self.order()                    # raises on a cycle

    def order(self) -> list[Node]:
        """Nodes in topological order, so a consumer wires after its source.

        Deterministic: ties break by name, so the same netlist always orders
        the same way and generated output diffs stay quiet.
        """
        incoming: dict[str, set] = {node.name: set() for node in self.nodes}
        for source, sink in self._edges:
            if source.node is not None and sink.node is not None:
                incoming[sink.node].add(source.node)

        ordered: list[Node] = []
        remaining = dict(self._nodes)
        while remaining:
            ready = [name for name in remaining
                     if not (incoming[name] & set(remaining))]
            if not ready:
                raise ValueError(
                    f"{self.name}: the netlist has a cycle among "
                    f"{sorted(remaining)}; a streaming pipeline must be "
                    "feed-forward")
            for name in sorted(ready):
                ordered.append(remaining.pop(name))
        return ordered
