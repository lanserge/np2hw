# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Toolchains: a loaded design rendered for the tool that builds it.

Each toolchain makes only its own blocks and builds what the design says;
it decides nothing about the design. :mod:`np2hw.toolchain.vivado` renders
a design for AMD Vivado and holds the checks of what Vivado built.
"""
