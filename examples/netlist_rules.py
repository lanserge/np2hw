"""The netlist checks: every rule a streaming design can die by, refused early.

np2hw's handshake implies laws about the GRAPH -- who may drive whom, when an
output may fan out -- and compose() enforces them at emission. netlist.py is
the same law one stage earlier, as a structure an application builds and
validates before anything is generated. It exists because an application that
re-derives these rules keeps a copy of the emitter's law, and the fork rule
really was implemented twice before this module.

Each check here demonstrates one refusal, plus the two things that must be
ACCEPTED for the rules to be worth anything: a tap (any number of sinks fan
out free, because a sink never stalls) and a diamond that is NOT a cycle.

Run:  python examples/netlist_rules.py
"""
import sys

from np2hw.netlist import Endpoint, Netlist, Node


def build(edges, nodes=None, outputs=("out",)):
    net = Netlist("demo", external_inputs=("in",), external_outputs=outputs)
    for node in nodes or ():
        net.add(node)
    for source, sink in edges:
        net.connect(source, sink)
    return net


def refuses(label, why, net, expect):
    try:
        net.validate()
    except ValueError as error:
        ok = expect in str(error)
        print(f"  {label:<34} {'PASS' if ok else 'FAIL'}  ({why})")
        return ok
    print(f"  {label:<34} FAIL  validated a graph that is {why}")
    return False


def main():
    a = Node("a")
    b = Node("b")
    watch = Node("watch", inputs=("in",), outputs=())      # a SINK: never stalls

    print("netlist_rules:")
    checks = []

    checks.append(refuses(
        "undriven input", "a stream that never arrives",
        build([("in", "a.in"), ("a.out", "out")], [a, b]),
        "b.in is not connected"))

    checks.append(refuses(
        "two drivers on one input", "a short",
        build([("in", "a.in"), ("in", "b.in"),
               ("a.out", "out"), ("b.out", "a.in")], [a, b]),
        "exactly one driver"))

    # A pure two-node loop: each input has exactly ONE driver, so nothing else
    # fires first and the cycle check is what actually catches it.
    loop_a, loop_b = Node("la"), Node("lb")
    checks.append(refuses(
        "cycle", "unschedulable in a feed-forward pipeline",
        build([("la.out", "lb.in"), ("lb.out", "la.in")], [loop_a, loop_b],
              outputs=()),
        "cycle"))

    checks.append(refuses(
        "fork to two blocking consumers", "a deadlock under load",
        build([("in", "a.in"), ("a.out", "b.in"), ("a.out", "out")], [a, b]),
        "buffering fork element"))

    # A tap is FREE: the same fork, but the second consumer is a sink.
    tapped = build([("in", "a.in"), ("a.out", "out"), ("a.out", "watch.in")],
                   [a, watch])
    tapped.validate()
    ok = tapped.is_tap(Endpoint("watch", "in")) and not tapped.is_tap(Endpoint(None, "out"))
    print(f"  {'tap to a sink is accepted':<34} {'PASS' if ok else 'FAIL'}  "
          "(a sink never stalls; an external output does)")
    checks.append(ok)

    # A diamond is not a cycle, and order() is deterministic.
    d1, d2, d3, join = Node("d1"), Node("d2", outputs=()), Node("d3", outputs=()), None
    diamond = build([("in", "d1.in"), ("d1.out", "d2.in"), ("d1.out", "d3.in"),
                     ("d1.out", "out")], [d1, d2, d3])
    diamond.validate()
    names = [n.name for n in diamond.order()]
    ok = names == ["d1", "d2", "d3"]
    print(f"  {'diamond orders deterministically':<34} {'PASS' if ok else 'FAIL'}  {names}")
    checks.append(ok)

    print("\n" + ("NETLIST PASS" if all(checks) else "FAIL"))
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
