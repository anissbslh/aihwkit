# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Analog Information utility.

This module prints relevant information about the model and its analog
execution.

Enhancements (issue #316 follow-ups):
- More summary info: remaining digital-MACs, total analog tile mat-vecs
- User-definable columns
- Improved layout (no confusing placeholder entries)
- Optional accounting for peripheral digital ops (noise/bound mgmt, etc.) via configurable factors
- Optional counting of backward / update passes (for analog training)
- Optional estimation of peripheral ops from RPU settings
- Hooks for analog optimizer specifics (tiki-taka / mixed precision) to account for extra internal tiles / ops

Notes:
- MAC / mat-vec counts are estimates based on tensor shapes observed in a dummy forward pass.
- Peripheral ops and optimizer-specific overhead are model/config dependent: exposed via user parameters/callbacks.

Example:
    >>> import torch
    >>> from torch import nn
    >>> from aihwkit.nn import AnalogLinear
    >>> from aihwkit.simulator.configs.configs import UnitCellRPUConfig
    >>> from aihwkit.simulator.configs.compounds import TransferCompound
    >>> from aihwkit.utils.analog_info import analog_summary
    >>>
    >>> rpu_config = UnitCellRPUConfig(device=TransferCompound())
    >>> model = nn.Sequential(
    ...     AnalogLinear(128, 64, rpu_config=rpu_config),
    ...     nn.ReLU(),
    ...     nn.Linear(64, 10),
    ... )
    >>> info = analog_summary(
    ...     model,
    ...     input_size=(1, 128),
    ...     rpu_config=rpu_config,
    ...     include_backward=True,
    ...     include_update=True,
    ...     estimate_peripheral_ops=True,
    ...     estimate_optimizer_overhead=True,
    ...     peripheral_ops_per_matvec=2,
    ... )
    >>> print(info)
"""

from functools import reduce
import operator
from typing import Optional, Any, List, Dict, Tuple, Callable

# Imports from PyTorch.
from torch import zeros
from torch.nn import Module

from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.nn.modules.conv_mapped import _AnalogConvNdMapped
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped
from aihwkit.nn.modules.conv import _AnalogConvNd
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.parameters.base import RPUConfigBase
from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType
from aihwkit.simulator.configs.compounds import MixedPrecisionCompound, TransferCompound
from aihwkit.simulator.configs.configs import DigitalRankUpdateRPUConfig, UnitCellRPUConfig


try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich import box

    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    _RICH_AVAILABLE = False


# -----------------------------
# Column configuration
# -----------------------------
COLUMN_DEFINITIONS = ["Layer Information", "Tile Information", "Compute Information"]

# key -> (section_index, header_name)
COLUMN_NAMES: Dict[str, Tuple[int, str]] = {
    # Layer
    "name": (0, "Layer Name"),
    "isanalog": (0, "Is Analog"),
    "input_size": (0, "In Shape"),
    "output_size": (0, "Out Shape"),
    "kernel_size": (0, "Kernel Shape"),
    "num_tiles": (0, "# of Tiles"),
    "reuse_factor": (0, "Reuse Factor"),
    # Tile
    "log_shape": (1, "Log. tile shape"),
    "phy_shape": (1, "Phys. tile shape"),
    "utilization": (1, "Util (%)"),
    # Compute
    "digital_macs": (2, "Digital MACs"),
    "analog_matvecs": (2, "Analog mat-vecs"),
    "peripheral_ops": (2, "Peripheral ops"),
}

DEFAULT_COLUMNS = [
    "name",
    "isanalog",
    "input_size",
    "output_size",
    "kernel_size",
    "num_tiles",
    "reuse_factor",
    "log_shape",
    "phy_shape",
    "utilization",
    "digital_macs",
    "analog_matvecs",
    "peripheral_ops",
]

FORMATTING_WIDTH = 200
COLUMN_WIDTH = 20
FLOAT_FORMAT = "{0:.2f}"


def _prod(vals: Any) -> int:
    return int(reduce(operator.mul, vals, 1))


def _fmt_int(x: Optional[int]) -> str:
    if x is None:
        return "-"
    return str(int(x))


def _fmt_float(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return FLOAT_FORMAT.format(float(x))


# -----------------------------
# TileInfo
# -----------------------------
class TileInfo:
    """Class for storing tile statistics and information."""

    log_in_size: Any
    log_out_size: Any
    phy_in_size: Any
    phy_out_size: Any
    utilization: float

    def __init__(self, tile: TileModule, is_mapped: bool):
        self.log_in_size = tile.in_size
        self.log_out_size = tile.out_size
        self.phy_in_size = tile.rpu_config.mapping.max_input_size
        self.phy_out_size = tile.rpu_config.mapping.max_output_size
        self.is_mapped = is_mapped
        max_space = self.phy_in_size * self.phy_out_size
        log_space = self.log_in_size * self.log_out_size
        self.utilization = log_space * 100 / max_space if is_mapped else 100.0

    def tile_summary_dict(self) -> dict:
        """Return a dictionary with the tile info."""
        phys_shape = "N/A" if not self.is_mapped else (self.phy_out_size, self.phy_in_size)
        return {
            "log_shape": str((self.log_out_size, self.log_in_size)),
            "phy_shape": str(phys_shape),
            "utilization": self.utilization,
        }


# -----------------------------
# LayerInfo
# -----------------------------
class LayerInfo:
    """Class for storing layer statistics and information."""

    # pylint: disable=too-many-instance-attributes
    module: Module
    name: str
    isanalog: bool
    num_tiles: int
    tiles_info: List[TileInfo]
    input_size: Any
    output_size: Any
    kernel_size: Any
    reuse_factor: int

    # Compute counts (estimates)
    digital_macs: int
    analog_matvecs: int
    peripheral_ops: int

    def __init__(
        self,
        module: Module,
        rpu_config: Optional[RPUConfigBase] = None,
        input_size: Any = None,
        output_size: Any = None,
        *,
        include_backward: bool = False,
        include_update: bool = False,
        peripheral_ops_per_matvec: int = 0,
        estimate_peripheral_ops: bool = False,
        estimate_optimizer_overhead: bool = False,
        extra_tiles_fn: Optional[Callable[[Module, Optional[RPUConfigBase]], int]] = None,
        extra_digital_ops_fn: Optional[Callable[[Module, Optional[RPUConfigBase]], int]] = None,
    ):
        self.module = module
        self.name = self.module.__class__.__name__
        self.isanalog = isinstance(self.module, AnalogLayerBase)

        self.include_backward = include_backward
        self.include_update = include_update
        self.rpu_config = rpu_config

        base_tiles = 0 if not self.isanalog else len(list(self.module.analog_tiles()))
        optimizer_extra_tiles, optimizer_extra_ops = self._estimate_optimizer_overhead(
            base_tiles, estimate_optimizer_overhead
        )
        extra_tiles = extra_tiles_fn(self.module, rpu_config) if extra_tiles_fn else 0
        self.num_tiles = base_tiles + int(optimizer_extra_tiles) + int(extra_tiles)

        self.tiles_info = self.set_tiles_info()
        self.kernel_size = None
        self.reuse_factor = 0
        self.input_size, self.output_size = input_size, output_size
        self.set_kernel_size()
        self.calculate_reuse_factor()

        # Pass factors: forward always counted once
        pass_factor = 1 + (1 if include_backward else 0) + (1 if include_update else 0)

        # Compute estimates
        self.digital_macs = self._estimate_digital_macs() * pass_factor if not self.isanalog else 0
        self.analog_matvecs = self._estimate_analog_matvecs() * pass_factor if self.isanalog else 0

        base_periph = int(peripheral_ops_per_matvec) * self.analog_matvecs
        estimated_periph = (
            self._estimate_peripheral_ops() if estimate_peripheral_ops and self.isanalog else 0
        )
        extra_digital = extra_digital_ops_fn(self.module, rpu_config) if extra_digital_ops_fn else 0
        extra_digital += optimizer_extra_ops
        # peripheral ops are digital-ish overhead around analog compute
        self.peripheral_ops = int(base_periph + estimated_periph + extra_digital) if self.isanalog else 0

    def set_tiles_info(self) -> List[TileInfo]:
        """Create TileInfo objects for each tile of the layer."""
        tiles_info: List[TileInfo] = []
        is_mapped = isinstance(self.module, AnalogLinearMapped)
        is_mapped = is_mapped or isinstance(self.module, _AnalogConvNdMapped)
        if isinstance(self.module, AnalogLayerBase):
            for tile in self.module.analog_tiles():
                tiles_info.append(TileInfo(tile, is_mapped))
        return tiles_info

    def set_kernel_size(self) -> None:
        """Set kernel size attribute."""
        if hasattr(self.module, "kernel_size"):
            self.kernel_size = self.module.kernel_size

    def calculate_reuse_factor(self) -> None:
        """Compute the reuse factor.

        The reuse factor is the number of vector matrix multiplication
        a layer computes (per forward).
        """
        if self.input_size is None or self.output_size is None:
            self.reuse_factor = 0
            return

        if isinstance(self.module, (_AnalogConvNd, _AnalogConvNdMapped)):
            ruf = _prod(self.output_size) // int(self.output_size[1])
            self.reuse_factor = int(ruf)
        elif isinstance(self.module, (AnalogLinear, AnalogLinearMapped)):
            ruf = _prod(self.input_size) // int(self.input_size[-1])
            self.reuse_factor = int(ruf)

    def _estimate_vector_lengths(self) -> Tuple[int, int]:
        """Estimate input/output vector lengths for a single mat-vec."""
        # Linear-like
        if hasattr(self.module, "in_features") and hasattr(self.module, "out_features"):
            in_f = int(getattr(self.module, "in_features"))
            out_f = int(getattr(self.module, "out_features"))
            return in_f, out_f

        # Conv-like
        if hasattr(self.module, "kernel_size") and hasattr(self.module, "in_channels") and hasattr(
            self.module, "out_channels"
        ):
            cin = int(getattr(self.module, "in_channels"))
            cout = int(getattr(self.module, "out_channels"))
            k = getattr(self.module, "kernel_size")
            if isinstance(k, int):
                k_elems = int(k * k)
            else:
                k_elems = int(_prod(list(k)))
            return cin * k_elems, cout

        return 0, 0

    def _estimate_noise_mgmt_ops(self, input_len: int, nm_type: NoiseManagementType) -> int:
        """Estimate digital ops for noise management per mat-vec."""
        if nm_type in (NoiseManagementType.NONE, NoiseManagementType.CONSTANT):
            return 0
        if nm_type == NoiseManagementType.AVERAGE_ABS_MAX:
            return int(2 * input_len + 1)
        return int(2 * input_len)

    def _estimate_bound_mgmt_ops(
        self, input_len: int, output_len: int, bm_type: BoundManagementType
    ) -> int:
        """Estimate digital ops for bound management per mat-vec."""
        if bm_type == BoundManagementType.NONE:
            return 0
        if bm_type == BoundManagementType.SHIFT:
            return int(output_len)
        if bm_type == BoundManagementType.ITERATIVE_WORST_CASE:
            return int(2 * (input_len + output_len))
        return int(input_len + output_len)

    def _estimate_peripheral_ops(self) -> int:
        """Estimate peripheral digital ops for forward/backward/update passes."""
        if self.reuse_factor == 0 or self.rpu_config is None:
            return 0

        input_len, output_len = self._estimate_vector_lengths()
        if input_len == 0 and output_len == 0:
            return 0

        ops = 0
        # Forward pass
        ops += self._estimate_noise_mgmt_ops(
            input_len, self.rpu_config.forward.noise_management
        )
        ops += self._estimate_bound_mgmt_ops(
            input_len, output_len, self.rpu_config.forward.bound_management
        )

        # Backward pass (if requested)
        if self.include_backward:
            ops += self._estimate_noise_mgmt_ops(
                output_len, self.rpu_config.backward.noise_management
            )
            ops += self._estimate_bound_mgmt_ops(
                output_len, input_len, self.rpu_config.backward.bound_management
            )

        # Per mat-vec ops -> scale by reuse and tile count
        total = ops * self.reuse_factor * self.num_tiles
        return int(total)

    def _estimate_optimizer_overhead(
        self, base_tiles: int, estimate_optimizer_overhead: bool
    ) -> Tuple[int, int]:
        """Estimate extra tiles and digital ops for optimizer-specific overhead."""
        if not estimate_optimizer_overhead or not self.isanalog or self.rpu_config is None:
            return 0, 0

        input_len, output_len = self._estimate_vector_lengths()
        extra_tiles = 0
        extra_ops = 0

        if isinstance(self.rpu_config, UnitCellRPUConfig) and isinstance(
            self.rpu_config.device, TransferCompound
        ):
            n_devices = len(self.rpu_config.device.unit_cell_devices)
            extra_tiles = max(0, n_devices - 1) * base_tiles
            if self.include_update and self.rpu_config.device.transfer_every > 0:
                transfers = self.reuse_factor / float(self.rpu_config.device.transfer_every)
                read_len = input_len if self.rpu_config.device.transfer_columns else output_len
                extra_ops = int(
                    transfers * self.rpu_config.device.n_reads_per_transfer * (read_len + output_len)
                )

        if isinstance(self.rpu_config, DigitalRankUpdateRPUConfig) and isinstance(
            self.rpu_config.device, MixedPrecisionCompound
        ):
            extra_tiles = max(extra_tiles, base_tiles)
            if self.include_update:
                transfer_every = max(1, int(self.rpu_config.device.transfer_every))
                extra_ops += int(self.reuse_factor * input_len * output_len / transfer_every)

        return int(extra_tiles), int(extra_ops)


    def _estimate_digital_macs(self) -> int:
        """Estimate MACs for digital layers from observed shapes."""
        if self.input_size is None or self.output_size is None or self.reuse_factor == 0:
            return 0

        # Linear-like
        if hasattr(self.module, "in_features") and hasattr(self.module, "out_features"):
            in_f = int(getattr(self.module, "in_features"))
            out_f = int(getattr(self.module, "out_features"))
            # Each matmul: out_f * in_f MACs; repeated reuse_factor times
            return int(self.reuse_factor * out_f * in_f)

        # Conv-like
        if hasattr(self.module, "kernel_size") and hasattr(self.module, "in_channels") and hasattr(
            self.module, "out_channels"
        ):
            cin = int(getattr(self.module, "in_channels"))
            cout = int(getattr(self.module, "out_channels"))
            k = getattr(self.module, "kernel_size")
            if isinstance(k, int):
                k_elems = int(k * k)
            else:
                k_elems = int(_prod(list(k)))
            # Each output element: cin * k_elems MACs; total output elements: reuse_factor * cout
            return int(self.reuse_factor * cout * cin * k_elems)

        # Unknown digital layer type
        return 0

    def _estimate_analog_matvecs(self) -> int:
        """Estimate analog tile mat-vec count (forward) from reuse_factor and tile count."""
        if self.reuse_factor == 0:
            return 0
        # Each tile performs one mat-vec per reuse factor (per forward)
        return int(self.num_tiles * self.reuse_factor)

    def layer_summary_dict(self) -> dict:
        """Return a dictionary with all layer's information."""
        analog = "analog" if self.isanalog else "digital"
        return {
            "name": self.name,
            "isanalog": analog,
            "input_size": str(self.input_size) if self.input_size is not None else "-",
            "output_size": str(self.output_size) if self.output_size is not None else "-",
            "kernel_size": str(self.kernel_size) if self.kernel_size is not None else "-",
            "num_tiles": self.num_tiles,
            "reuse_factor": str(self.reuse_factor) if self.reuse_factor is not None else "-",
            # Tile columns shown only on tile rows
            "log_shape": "",
            "phy_shape": "",
            "utilization": "",
            # Compute columns
            "digital_macs": self.digital_macs,
            "analog_matvecs": self.analog_matvecs,
            "peripheral_ops": self.peripheral_ops,
        }


# -----------------------------
# AnalogInfo
# -----------------------------
class AnalogInfo:
    """Class for computing and storing results of the analog summary."""

    def __init__(
        self,
        model: Module,
        input_size: Any = None,
        rpu_config: Optional[RPUConfigBase] = None,
        *,
        columns: Optional[List[str]] = None,
        include_backward: bool = False,
        include_update: bool = False,
        peripheral_ops_per_matvec: int = 0,
        estimate_peripheral_ops: bool = False,
        estimate_optimizer_overhead: bool = False,
        extra_tiles_fn: Optional[Callable[[Module, Optional[RPUConfigBase]], int]] = None,
        extra_digital_ops_fn: Optional[Callable[[Module, Optional[RPUConfigBase]], int]] = None,
    ):
        self.model = model
        self.input_size = input_size
        self.rpu_config = rpu_config
        self.columns = columns if columns is not None else list(DEFAULT_COLUMNS)

        # Validate columns (ignore unknowns silently)
        self.columns = [c for c in self.columns if c in COLUMN_NAMES]

        self.include_backward = include_backward
        self.include_update = include_update
        self.peripheral_ops_per_matvec = peripheral_ops_per_matvec
        self.estimate_peripheral_ops = estimate_peripheral_ops
        self.estimate_optimizer_overhead = estimate_optimizer_overhead
        self.extra_tiles_fn = extra_tiles_fn
        self.extra_digital_ops_fn = extra_digital_ops_fn

        self.layer_summary = self.create_layer_summary()

        # Totals
        self.total_tile_number = sum(x.num_tiles for x in self.layer_summary)
        self.total_nb_analog = sum(1 for x in self.layer_summary if x.isanalog)
        self.total_digital_macs = sum(x.digital_macs for x in self.layer_summary)
        self.total_analog_matvecs = sum(x.analog_matvecs for x in self.layer_summary)
        self.total_peripheral_ops = sum(x.peripheral_ops for x in self.layer_summary)

    def register_hooks_recursively(self, module: Module, hook: Any) -> None:
        """Hooks the function into all layers with no children (or only analog tiles as children)."""
        if len(list(module.children())) == 0:
            module.register_forward_hook(hook)
        elif (
            isinstance(module, AnalogLayerBase)
            and not module.IS_CONTAINER
            and len([ch for ch in module.children() if isinstance(ch, AnalogLayerBase)]) == 0  # type: ignore
        ):
            module.register_forward_hook(hook)  # type: ignore
        else:
            for layer in module.children():
                self.register_hooks_recursively(layer, hook)

    def create_layer_summary(self) -> List[LayerInfo]:
        """Create the layer summary list."""
        layer_summary: List[LayerInfo] = []

        def get_size_hook(mod: Module, _input: Any, _output: Any) -> None:
            input_size = list(_input[0].size())
            output_size = list(_output.size())
            layer_summary.append(
                LayerInfo(
                    mod,
                    self.rpu_config,
                    input_size,
                    output_size,
                    include_backward=self.include_backward,
                    include_update=self.include_update,
                    peripheral_ops_per_matvec=self.peripheral_ops_per_matvec,
                    estimate_peripheral_ops=self.estimate_peripheral_ops,
                    estimate_optimizer_overhead=self.estimate_optimizer_overhead,
                    extra_tiles_fn=self.extra_tiles_fn,
                    extra_digital_ops_fn=self.extra_digital_ops_fn,
                )
            )

        self.register_hooks_recursively(self.model, get_size_hook)
        device = next(self.model.parameters()).device
        dummy_var = zeros(self.input_size).to(device)
        self.model(dummy_var)
        return layer_summary

    def _header_by_section(self) -> Dict[int, List[str]]:
        by_sec: Dict[int, List[str]] = {0: [], 1: [], 2: []}
        for c in self.columns:
            sec, title = COLUMN_NAMES[c]
            by_sec[sec].append(title)
        return by_sec

    def _keys_by_section(self) -> Dict[int, List[str]]:
        by_sec: Dict[int, List[str]] = {0: [], 1: [], 2: []}
        for c in self.columns:
            sec, _ = COLUMN_NAMES[c]
            by_sec[sec].append(c)
        return by_sec

    # -----------------------------
    # Rich output (pretty table)
    # -----------------------------
    def to_rich(
        self,
        *,
        show_box: bool = True,
        title: Optional[str] = None,
        show_lines: bool = False,
        zebra: bool = True,
    ):
        """Create Rich renderables (table + summary panel).

        Requires: `rich` installed.
        """
        if not _RICH_AVAILABLE:
            raise RuntimeError(
                "Rich is not available. Install it with: pip install rich "
                "or use the default string output (print(info))."
            )

        # Build column list in the exact order user requested
        ordered_cols: List[Tuple[int, str, str]] = []  # (section_idx, key, header_title)
        for c in self.columns:
            sec, h = COLUMN_NAMES[c]
            ordered_cols.append((sec, c, h))

        table_title = title or f"Analog Summary — {self.model.__class__.__name__}"
        table = Table(
            title=table_title,
            show_header=True,
            header_style="bold",
            show_lines=show_lines,
            box=box.SIMPLE_HEAVY if show_box else None,
            expand=True,
            row_styles=("none", "dim") if zebra else None,
        )

        # Add columns with alignment and section coloring
        for sec, key, header in ordered_cols:
            justify = "left"
            if key in ("num_tiles", "reuse_factor", "digital_macs", "analog_matvecs", "peripheral_ops"):
                justify = "right"
            elif key in ("utilization",):
                justify = "right"

            style = None
            if sec == 0:
                style = "white"
            elif sec == 1:
                style = "cyan"
            elif sec == 2:
                style = "magenta"

            table.add_column(header, justify=justify, style=style, no_wrap=False)

        def fmt_cell(k: str, v: Any) -> str:
            if v is None:
                return "-"
            if k in ("digital_macs", "analog_matvecs", "peripheral_ops"):
                return _fmt_int(int(v)) if isinstance(v, int) else "-"
            if k == "utilization":
                try:
                    return _fmt_float(float(v))
                except Exception:
                    return "-"
            return str(v)

        show_tile_cols = any(k in ("log_shape", "phy_shape", "utilization") for k in self.columns)

        # Add rows
        for layer in self.layer_summary:
            d = layer.layer_summary_dict()

            # Layer row
            row = []
            for _sec, k, _h in ordered_cols:
                if k in ("log_shape", "phy_shape", "utilization"):
                    row.append("-")
                else:
                    row.append(fmt_cell(k, d.get(k)))

            # Highlight analog layer rows slightly
            layer_style = "bold" if layer.isanalog else None
            table.add_row(*row, style=layer_style)

            # Tile rows
            if layer.isanalog and show_tile_cols and layer.tiles_info:
                for tile in layer.tiles_info:
                    td = tile.tile_summary_dict()
                    tile_row = []
                    for _sec, k, _h in ordered_cols:
                        if k == "name":
                            tile_row.append("  ↳ tile")
                        elif k == "isanalog":
                            tile_row.append("")
                        elif k in ("log_shape", "phy_shape", "utilization"):
                            tile_row.append(fmt_cell(k, td.get(k)))
                        else:
                            tile_row.append("")
                    table.add_row(*tile_row, style="dim")

        passes = ["forward"]
        if self.include_backward:
            passes.append("backward")
        if self.include_update:
            passes.append("update")

        summary_text = Text()
        summary_text.append(f"Passes counted: {', '.join(passes)}\n", style="bold")
        summary_text.append(f"Total number of tiles: {self.total_tile_number}\n")
        summary_text.append(f"Total number of analog layers: {self.total_nb_analog}\n")
        summary_text.append(f"Remaining digital MACs: {self.total_digital_macs}\n")
        summary_text.append(f"Total analog tile mat-vecs: {self.total_analog_matvecs}\n")
        summary_text.append(f"Peripheral digital ops (estimated): {self.total_peripheral_ops}\n")

        summary_panel = Panel(summary_text, title="General Information", box=box.SIMPLE)
        return table, summary_panel

    def print_rich(
        self,
        *,
        console: Optional["Console"] = None,
        show_box: bool = True,
        title: Optional[str] = None,
        show_lines: bool = False,
        zebra: bool = False,
    ) -> None:
        """Pretty-print using Rich."""
        if not _RICH_AVAILABLE:
            # Fall back to classic output
            print(self)
            return

        c = console or Console()
        table, summary = self.to_rich(show_box=show_box, title=title, show_lines=show_lines, zebra=zebra)
        c.print(table)
        c.print(summary)

    # -----------------------------
    # Existing ASCII output
    # -----------------------------
    def __repr__(self) -> str:
        divider = "=" * FORMATTING_WIDTH + "\n"
        name = "Model Name: " + self.model.__class__.__name__ + "\n"
        passes = "Passes counted: forward"
        if self.include_backward:
            passes += " + backward"
        if self.include_update:
            passes += " + update"
        passes += "\n"

        result = divider + name + passes + divider
        result += "Per-layer Information\n" + divider

        headers = self._header_by_section()
        keys = self._keys_by_section()

        # Section titles row
        for i, category in enumerate(COLUMN_DEFINITIONS):
            if len(headers[i]) == 0:
                continue
            trim_length = COLUMN_WIDTH * len(headers[i]) - len(category)
            result += category + " " * max(1, trim_length)
            if i != len(COLUMN_DEFINITIONS) - 1:
                result += "| "
        result += "\n" + divider

        # Column headers row
        header_line = ""
        for i in range(len(COLUMN_DEFINITIONS)):
            if len(headers[i]) == 0:
                continue
            header_line += ("{:<" + str(COLUMN_WIDTH) + "}") * len(headers[i])
        result += header_line.format(*(headers[0] + headers[1] + headers[2])) + "\n"

        # Rows
        for layer in self.layer_summary:
            d = layer.layer_summary_dict()

            # Layer row
            row_vals: List[str] = []
            for sec in range(len(COLUMN_DEFINITIONS)):
                for k in keys[sec]:
                    if k in ("log_shape", "phy_shape", "utilization"):
                        row_vals.append("-")
                    elif k in ("digital_macs", "analog_matvecs", "peripheral_ops"):
                        row_vals.append(_fmt_int(int(d[k])) if isinstance(d[k], int) else "-")
                    else:
                        row_vals.append(str(d[k]))

            row_fmt = ("{:<" + str(COLUMN_WIDTH) + "}") * len(row_vals)
            result += row_fmt.format(*row_vals) + "\n"

            # Tile rows
            show_tile_cols = any(k in ("log_shape", "phy_shape", "utilization") for k in self.columns)
            if layer.isanalog and show_tile_cols and layer.tiles_info:
                for tile in layer.tiles_info:
                    td = tile.tile_summary_dict()
                    tile_vals: List[str] = []
                    for sec in range(len(COLUMN_DEFINITIONS)):
                        for k in keys[sec]:
                            if k == "log_shape":
                                tile_vals.append(str(td["log_shape"]))
                            elif k == "phy_shape":
                                tile_vals.append(str(td["phy_shape"]))
                            elif k == "utilization":
                                tile_vals.append(_fmt_float(float(td["utilization"])))
                            else:
                                tile_vals.append("")
                    result += row_fmt.format(*tile_vals) + "\n"

        result += divider
        result += "General Information\n" + divider
        result += "Total number of tiles: " + str(self.total_tile_number) + "\n"
        result += "Total number of analog layers: " + str(self.total_nb_analog) + "\n"
        result += "Remaining digital MACs: " + str(self.total_digital_macs) + "\n"
        result += "Total analog tile mat-vecs: " + str(self.total_analog_matvecs) + "\n"
        result += "Peripheral digital ops (estimated): " + str(self.total_peripheral_ops) + "\n"
        return result


def analog_summary(
    model: Module,
    input_size: Optional[Any] = None,
    rpu_config: Optional[RPUConfigBase] = None,
    *,
    columns: Optional[List[str]] = None,
    include_backward: bool = False,
    include_update: bool = False,
    peripheral_ops_per_matvec: int = 0,
    estimate_peripheral_ops: bool = False,
    estimate_optimizer_overhead: bool = False,
    extra_tiles_fn: Optional[Callable[[Module, Optional[RPUConfigBase]], int]] = None,
    extra_digital_ops_fn: Optional[Callable[[Module, Optional[RPUConfigBase]], int]] = None,
) -> AnalogInfo:
    """Summarize the given PyTorch model.

    Enhancements:
      - remaining digital MACs
      - total analog tile mat-vecs
      - user-definable columns
      - optional backward/update counting
      - optional peripheral ops accounting (configurable)
      - optional optimizer overhead hooks (extra tiles / extra digital ops)

    Args:
        model: PyTorch model.
        input_size: required to run a forward pass of the model.
        rpu_config: resistive processing unit configuration.

        columns: list of column keys to display (see COLUMN_NAMES keys).
        include_backward: if True, counts are scaled to include backward pass.
        include_update: if True, counts are scaled to include update pass.
        peripheral_ops_per_matvec: estimate of extra digital ops per analog mat-vec.
        estimate_peripheral_ops: if True, estimates per-matvec peripheral ops from RPU settings.
        estimate_optimizer_overhead: if True, estimates optimizer-specific extra tiles/ops.
        extra_tiles_fn: callback(module, rpu_config)->int for optimizer internals (e.g. tiki-taka).
        extra_digital_ops_fn: callback(module, rpu_config)->int for optimizer/peripheral overhead.

    Returns:
        AnalogInfo Object.
    """
    return AnalogInfo(
        model,
        input_size,
        rpu_config,
        columns=columns,
        include_backward=include_backward,
        include_update=include_update,
        peripheral_ops_per_matvec=peripheral_ops_per_matvec,
        estimate_peripheral_ops=estimate_peripheral_ops,
        estimate_optimizer_overhead=estimate_optimizer_overhead,
        extra_tiles_fn=extra_tiles_fn,
        extra_digital_ops_fn=extra_digital_ops_fn,
    )
