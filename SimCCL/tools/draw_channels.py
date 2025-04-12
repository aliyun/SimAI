import plotly.graph_objects as go
import plotly.io as pio

# Channel data (add more channels as needed)
channels = [
    [0, 7, 6, 5, 4, 3, 2, 1, 9, 10, 11, 12, 13, 14, 15, 8, 16, 23, 22, 21, 20, 19, 18, 17, 25, 26, 27, 28, 29, 30, 31, 24, 0],
    [0, 8, 15, 14, 13, 12, 11, 10, 9, 17, 18, 19, 20, 21, 22, 23, 16, 24, 31, 30, 29, 28, 27, 26, 25, 1, 2, 3, 4, 5, 6, 7, 0],
]

# GPU information
gpu_info = {}
for i in range(32):
    gpu_info[i] = {"name": f"GPU{i}", "compute_power": 10 + i % 5}

# GPU positions
gpu_positions = {
    0: (0, 1), 7: (0, 0), 1: (1, 1), 6: (1, 0),
    2: (2, 1), 5: (2, 0), 3: (3, 1), 4: (3, 0),
    8: (0, 3), 15: (0, 2), 9: (1, 3), 14: (1, 2),
    10: (2, 3), 13: (2, 2), 11: (3, 3), 12: (3, 2),
    16: (4, 1), 23: (4, 0), 17: (5, 1), 22: (5, 0),
    18: (6, 1), 21: (6, 0), 19: (7, 1), 20: (7, 0),
    24: (4, 3), 31: (4, 2), 25: (5, 3), 30: (5, 2),
    26: (6, 3), 29: (6, 2), 27: (7, 3), 28: (7, 2),
}

# AI conference color scheme (example)
colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]

def create_channel_trace(channel_data, channel_index):
    """Creates a trace for a single channel with directional arrows."""
    x_coords = []
    y_coords = []
    for i in range(len(channel_data) - 1):
        start_gpu = channel_data[i]
        end_gpu = channel_data[i + 1]
        x_coords.extend([gpu_positions[start_gpu][0], gpu_positions[end_gpu][0], None])
        y_coords.extend([gpu_positions[start_gpu][1], gpu_positions[end_gpu][1], None])

    trace = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="lines",  # Changed to just lines, as arrows will be handled separately
        name=f"Channel {channel_index}",
        hovertemplate="From: GPU %{text[0]}<br>To: GPU %{text[1]}<br>Communication Type: %{customdata}",
        text=[[gpu_info[channel_data[i]]["name"], gpu_info[channel_data[i + 1]]["name"]] for i in
              range(len(channel_data) - 1)],
        customdata=["NCCL" for _ in range(len(channel_data) - 1)],
        line=dict(color=colors[channel_index % len(colors)], width=2),
        visible=True,  # Show all channels initially
        # visible=(channel_index == 0)
    )
    return trace

def create_gpu_nodes():
    """Creates scatter points for GPU nodes with labels."""
    x_coords = [pos[0] for pos in gpu_positions.values()]
    y_coords = [pos[1] for pos in gpu_positions.values()]

    trace = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers+text",  # Add text for labels
        name="GPUs",
        hovertemplate="GPU: %{text}<br>Compute Power: %{customdata}",
        text=[gpu_info[gpu]["name"] for gpu in gpu_positions.keys()],
        customdata=[gpu_info[gpu]["compute_power"] for gpu in gpu_positions.keys()],
        marker=dict(size=20, symbol="circle-open", color="green", line=dict(width=2)),
        textposition="top right",  # Position labels
        textfont=dict(size=10),
    )
    return trace

def create_arrow_annotations(channels):
    """Creates arrow annotations for all channels."""
    annotations = []
    for channel_index, channel in enumerate(channels):
        for i in range(len(channel) - 1):
            start_gpu = channel[i]
            end_gpu = channel[i + 1]
            start_pos = gpu_positions[start_gpu]
            end_pos = gpu_positions[end_gpu]
            
            # Calculate the point for the arrow closer to end_pos (4/5 of the way)
            arrow_x = start_pos[0] + 95 * (end_pos[0] - start_pos[0]) / 100
            arrow_y = start_pos[1] + 95 * (end_pos[1] - start_pos[1]) / 100
            
            annotations.append(
                dict(
                    x=arrow_x,
                    y=arrow_y,
                    ax=start_pos[0],
                    ay=start_pos[1],
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,  # Increased arrow size
                    arrowwidth=2,
                    arrowcolor=colors[channel_index % len(colors)],
                    visible=True,
                    # visible=(channel_index == 0)
                )
            )
    return annotations

# Create traces
traces = [create_channel_trace(channel, i) for i, channel in enumerate(channels)]
traces.append(create_gpu_nodes())

# Create arrow annotations
arrow_annotations = create_arrow_annotations(channels)

# Create layout
layout = go.Layout(
    title="NCCL Channel Graph",
    title_x=0.5,
    xaxis_title="GPU X Position",
    yaxis_title="GPU Y Position",
    yaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=1
    ),
    updatemenus=[
        dict(
            buttons=[
                dict(
                    args=[
                        {"visible": [True for _ in range(len(channels))] + [True]},
                        {"annotations": arrow_annotations}
                    ],
                    label="All Channels",
                    method="update",
                )
            ] + [
                dict(
                    args=[
                        {"visible": [(i == j) for j in range(len(channels))] + [True]},
                        {"annotations": [ann for ann in arrow_annotations if colors[i % len(colors)] == ann["arrowcolor"]]}
                    ],
                    label=f"Channel {i}",
                    method="update",
                )
                for i in range(len(channels))
            ],
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.05,
            yanchor="top",
        ),
    ],
    annotations=arrow_annotations
)

# Create figure
fig = go.Figure(data=traces, layout=layout)

# Save as HTML
pio.write_html(fig, file="nccl_channel_graph_example.html", auto_open=True)