# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
#
# np2hw_cdc_fifo's timing intent, SCOPED to the module (Vivado:
# set_property SCOPED_TO_REF np2hw_cdc_fifo, PROCESSING_ORDER LATE), so it
# applies to every instance by the module's own names. docs/clock-crossings.md
# says why. The two gray-coded pointers are synchronised by construction,
# so each is timed datapath-only -- and bounded by one period of the clock
# that SAMPLES it, looked up through the sampling registers, where a clock
# propagates: gray coding tolerates a sample caught mid-transition on ONE
# bit, and skew between a pointer's bits beyond a destination period breaks
# that. Nothing else crosses: the memory is block RAM, read by the reader's
# clock.
set_max_delay -datapath_only \
    -from [get_cells {rgray_reg[*]}] -to [get_cells {wq_rgray0_reg[*]}] \
    [get_property -min PERIOD [get_clocks -of_objects [get_cells {wq_rgray0_reg[*]}]]]
set_max_delay -datapath_only \
    -from [get_cells {wgray_reg[*]}] -to [get_cells {rq_wgray0_reg[*]}] \
    [get_property -min PERIOD [get_clocks -of_objects [get_cells {rq_wgray0_reg[*]}]]]
