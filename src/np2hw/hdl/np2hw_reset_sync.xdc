# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
#
# np2hw_reset_sync's timing intent, SCOPED to the module (Vivado:
# set_property SCOPED_TO_REF np2hw_reset_sync, PROCESSING_ORDER LATE), so it
# applies to every instance by the module's own names.
#
# The reset's sources are asynchronous by design -- a link's reset on the
# cable's clock, a processor's reset on none -- and they reach every flop
# here only through the asynchronous clear and preset. Neither edge of that
# is a timing path: assertion may land at any moment, and release is what
# the chain resynchronises -- the first flop may go metastable and the
# stages after it absorb it, while every other flop lets go with a 0 at its
# input, so nothing it drives can glitch. Timed as a recovery check across
# two unrelated clocks, it would fail by whatever the clocks' phase happens
# to be, and mean nothing.
set_false_path -to [get_pins -of_objects [get_cells -filter {IS_SEQUENTIAL}] \
    -filter {IS_CLEAR || IS_PRESET}]
