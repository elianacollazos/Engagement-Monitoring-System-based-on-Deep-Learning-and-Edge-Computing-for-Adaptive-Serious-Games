import plotly.graph_objects as go

# ==========================================
# COLORS
# ==========================================

emotion_colors = {
    "Neutral": "gray",
    "Happy": "yellow",
    "Sad": "blue",
    "Surprise": "orange",
    "Fear": "purple",
    "Disgust": "green",
    "Anger": "red",
    "Contempt": "brown"
}

# ==========================================
# FUNCTION
# ==========================================

def create_circumplex_plot(history, show_trajectory=True):

    if history is None or len(history) == 0:
        return None

    valence = [float(h["valence"]) for h in history]
    arousal = [float(h["arousal"]) for h in history]
    emotions = [h["emotion"] for h in history]

    colors = [emotion_colors.get(e, "black") for e in emotions]

    fig = go.Figure()

    # ==========================================
    # HISTORY
    # ==========================================

    fig.add_trace(go.Scatter(
        x=valence,
        y=arousal,
        mode="markers",
        marker=dict(
            color=colors,
            size=9,
            opacity=0.7,
            line=dict(color="black", width=1)
        ),
        text=emotions,
        hovertemplate=
            "Emotion: %{text}<br>" +
            "Valence: %{x:.2f}<br>" +
            "Arousal: %{y:.2f}",
        name="History"
    ))

    # ==========================================
    # CURRENT POINT
    # ==========================================

    fig.add_trace(go.Scatter(
        x=[valence[-1]],
        y=[arousal[-1]],
        mode="markers+text",
        marker=dict(
            color=colors[-1],
            size=18,
            line=dict(color="black", width=2)
        ),
        text=[emotions[-1]],
        textposition="top center",
        name="Current"
    ))

    # ==========================================
    # TRAJECTORY
    # ==========================================

    if show_trajectory and len(valence) > 1:
        fig.add_trace(go.Scatter(
            x=valence,
            y=arousal,
            mode="lines",
            line=dict(
                color="rgba(0,0,0,0.3)",
                width=2,
                dash="dot"
            ),
            name="Trajectory"
        ))

    # ==========================================
# AXES
    # ==========================================

    fig.add_shape(type="line", x0=-1, x1=1, y0=0, y1=0,
                  line=dict(color="black", width=1))

    fig.add_shape(type="line", x0=0, x1=0, y0=-1, y1=1,
                  line=dict(color="black", width=1))

    # ==========================================
# QUADRANTS
    # ==========================================

    fig.add_annotation(x=0.7, y=0.7, text="High valence / High arousal", showarrow=False)
    fig.add_annotation(x=-0.7, y=0.7, text="Low valence / High arousal", showarrow=False)
    fig.add_annotation(x=-0.7, y=-0.7, text="Low valence / Low arousal", showarrow=False)
    fig.add_annotation(x=0.7, y=-0.7, text="High valence / Low arousal", showarrow=False)

    fig.update_layout(
        title="Valence-Arousal Circumplex",
        xaxis_title="Valence",
        yaxis_title="Arousal",
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1], scaleanchor="x", scaleratio=1),
        height=600
    )

    return fig
