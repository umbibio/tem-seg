from typing import Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def plot_unet_plusplus(
    layer_depth: int = 5,
    filters_root: int = 16,
    show_filters: bool = True,
    show_operations: bool = True,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Create a graphical representation of the UNet++ architecture.

    Args:
        layer_depth: Number of layers in the network
        filters_root: Base number of filters
        show_filters: Whether to show filter counts in nodes
        show_operations: Whether to show operation types on edges
        figsize: Figure size for the plot
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Initialize node positions and metadata
    nodes = {}  # (i, j) -> (x, y, filters)
    filter_list = [filters_root * 2**i for i in range(layer_depth)]

    # Define spacing
    x_spacing = 2.5
    y_spacing = 2.0
    node_width = 1.8
    node_height = 0.8

    # Position nodes
    for i in range(layer_depth):
        for j in range(layer_depth - i):
            x = j * x_spacing
            y = -i * y_spacing
            nodes[(i, j)] = {"x": x, "y": y, "filters": filter_list[i], "exists": False}

    # Define color scheme
    encoder_color = "#FFE6E6"  # Light red for encoder
    decoder_color = "#E6F3FF"  # Light blue for decoder
    dense_color = "#E6FFE6"  # Light green for dense blocks

    # Draw nodes and mark which exist in the architecture
    for i in range(layer_depth):
        # Encoder path (leftmost column)
        nodes[(i, 0)]["exists"] = True

    # Mark nested skip pathway nodes
    for i in range(layer_depth - 1):
        for j in range(1, layer_depth - i):
            nodes[(i, j)]["exists"] = True

    # Draw nodes
    for (i, j), node_info in nodes.items():
        if not node_info["exists"]:
            continue

        x, y = node_info["x"], node_info["y"]

        # Determine node color
        if j == 0:
            color = encoder_color
            node_type = "Encoder"
        elif i == 0:
            color = decoder_color
            node_type = "Output"
        else:
            color = dense_color
            node_type = "Dense"

        # Create fancy box for node
        box = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(box)

        # Add node label
        label = f"X[{i},{j}]"
        if show_filters:
            label += f"\n{node_info['filters']} filters"

        ax.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold")

        # Add node type indicator
        ax.text(
            x,
            y + node_height / 2 + 0.3,
            node_type,
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    # Draw connections
    arrow_props = dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", linewidth=1.5)

    # 1. Encoder downsampling connections (red arrows)
    for i in range(layer_depth - 1):
        if nodes[(i, 0)]["exists"] and nodes[(i + 1, 0)]["exists"]:
            x1, y1 = nodes[(i, 0)]["x"], nodes[(i, 0)]["y"]
            x2, y2 = nodes[(i + 1, 0)]["x"], nodes[(i + 1, 0)]["y"]

            arrow = FancyArrowPatch(
                (x1, y1 - node_height / 2),
                (x2, y2 + node_height / 2),
                **arrow_props,
                color="red",
                alpha=0.7,
            )
            ax.add_patch(arrow)

            if show_operations:
                mid_y = (y1 + y2) / 2
                ax.text(
                    x1 - 0.7,
                    mid_y,
                    "MaxPool\n+Conv",
                    fontsize=7,
                    ha="center",
                    color="red",
                    weight="bold",
                )

    # 2. Upsampling connections (blue arrows)
    for i in range(layer_depth - 1):
        for j in range(1, layer_depth - i):
            if nodes[(i + 1, j - 1)]["exists"] and nodes[(i, j)]["exists"]:
                x1, y1 = nodes[(i + 1, j - 1)]["x"], nodes[(i + 1, j - 1)]["y"]
                x2, y2 = nodes[(i, j)]["x"], nodes[(i, j)]["y"]

                arrow = FancyArrowPatch(
                    (x1 + node_width / 2, y1),
                    (x2 - node_width / 2, y2),
                    **arrow_props,
                    color="blue",
                    alpha=0.7,
                )
                ax.add_patch(arrow)

                if show_operations:
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(
                        mid_x,
                        mid_y + 0.2,
                        "UpConv",
                        fontsize=7,
                        ha="center",
                        color="blue",
                        weight="bold",
                    )

    # 3. Dense skip connections (green arrows)
    for i in range(layer_depth - 1):
        for j in range(1, layer_depth - i):
            # Connect all previous nodes at the same depth
            for k in range(j):
                if nodes[(i, k)]["exists"] and nodes[(i, j)]["exists"]:
                    x1, y1 = nodes[(i, k)]["x"], nodes[(i, k)]["y"]
                    x2, y2 = nodes[(i, j)]["x"], nodes[(i, j)]["y"]

                    # Create curved arrow for skip connections
                    arrow = FancyArrowPatch(
                        (x1 + node_width / 2, y1),
                        (x2 - node_width / 2, y2),
                        **{**arrow_props, "connectionstyle": "arc3,rad=0.3"},
                        color="green",
                        alpha=0.5,
                    )
                    ax.add_patch(arrow)

    # Add title and legend
    ax.set_title(
        f"UNet++ Architecture (Depth={layer_depth})", fontsize=16, weight="bold", pad=20
    )

    # Create legend
    legend_elements = [
        patches.Patch(color=encoder_color, label="Encoder blocks"),
        patches.Patch(color=dense_color, label="Dense blocks"),
        patches.Patch(color=decoder_color, label="Output blocks"),
        patches.FancyArrowPatch((0, 0), (1, 0), color="red", label="Downsampling"),
        patches.FancyArrowPatch((0, 0), (1, 0), color="blue", label="Upsampling"),
        patches.FancyArrowPatch(
            (0, 0), (1, 0), color="green", label="Skip connections"
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    # Add annotations
    ax.text(
        0,
        -layer_depth * y_spacing - 1,
        "Note: Each node represents a feature map. Numbers indicate filter counts.",
        fontsize=10,
        ha="left",
        style="italic",
    )

    # Set axis properties
    ax.set_xlim(-1, (layer_depth - 1) * x_spacing + 1)
    ax.set_ylim(-layer_depth * y_spacing - 1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_unet_plusplus_detailed(
    layer_depth: int = 4,
    filters_root: int = 16,
    input_size: Tuple[int, int] = (256, 256),
    show_sizes: bool = True,
) -> None:
    """
    Create a more detailed visualization showing tensor dimensions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Initialize node positions and metadata
    nodes = {}
    filter_list = [filters_root * 2**i for i in range(layer_depth)]
    size_list = [
        (input_size[0] // (2**i), input_size[1] // (2**i)) for i in range(layer_depth)
    ]

    # Define spacing
    x_spacing = 3.5
    y_spacing = 2.5
    node_width = 2.8
    node_height = 1.2

    # Position and create detailed nodes
    for i in range(layer_depth):
        for j in range(layer_depth - i):
            x = j * x_spacing
            y = -i * y_spacing

            # Determine if node exists in architecture
            exists = (j == 0) or (i < layer_depth - 1 and j < layer_depth - i)

            if exists:
                nodes[(i, j)] = {
                    "x": x,
                    "y": y,
                    "filters": filter_list[i],
                    "size": size_list[i],
                    "exists": True,
                }

    # Draw detailed nodes
    for (i, j), node_info in nodes.items():
        x, y = node_info["x"], node_info["y"]

        # Determine node style
        if j == 0:
            color = "#FFB6C1"  # Light pink for encoder
            node_type = "ENCODER"
        elif i == 0:
            color = "#87CEEB"  # Sky blue for output
            node_type = "OUTPUT"
        else:
            color = "#90EE90"  # Light green for dense
            node_type = "DENSE"

        # Create multi-line box
        box = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor="black",
            linewidth=2.5,
        )
        ax.add_patch(box)

        # Create detailed label
        label_lines = [
            f"X[{i},{j}]",
            f"{node_type}",
        ]
        if show_sizes:
            h, w = node_info["size"]
            label_lines.append(f"{h}×{w}×{node_info['filters']}")

        label = "\n".join(label_lines)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, weight="bold")

    # Draw all connections with labels
    # [Similar connection drawing code as above, but with more detailed labels]

    # Add computation flow indicators
    ax.annotate(
        "INPUT",
        xy=(0, 0.8),
        xytext=(-1.5, 0.8),
        arrowprops=dict(arrowstyle="->", lw=2),
        fontsize=12,
        weight="bold",
    )

    final_x = (layer_depth - 1) * x_spacing
    ax.annotate(
        "OUTPUT",
        xy=(final_x, 0),
        xytext=(final_x + 1.5, 0),
        arrowprops=dict(arrowstyle="->", lw=2),
        fontsize=12,
        weight="bold",
    )

    ax.set_title(
        f"UNet++ Detailed Architecture\nInput: {input_size[0]}×{input_size[1]}, Base Filters: {filters_root}",
        fontsize=16,
        weight="bold",
        pad=20,
    )

    ax.set_xlim(-2.5, (layer_depth - 1) * x_spacing + 2.5)
    ax.set_ylim(-layer_depth * y_spacing - 1, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.show()
