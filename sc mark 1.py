# sc_mark_1.py
# Half-court (vertical, hoop at bottom) with:
# - tiny connector from backboard to BACK of rim
# - half-court line + downward semicircle only
# - centered graph (origin (0,0) stays at left-baseline corner)
# - BOTH axes completely hidden (no ticks/labels)
# - y-axis hard-capped to 48 ft so nothing shows above half court

import math
import numpy as np
import dash
from dash import html, dcc, Output, Input, no_update
import plotly.graph_objects as go

# ---------------- Court geometry (feet) ----------------
COURT_W = 50.0          # x: left→right
HALF_H  = 47.0          # y: baseline (0) → half court (47)

# Basket & key
RIM_X   = 25.0
RIM_Y   = 5.25
RIM_R   = 0.75
BACKBOARD_Y = 4.0
RESTRICTED_R = 4.0

LANE_W  = 16.0
LANE_X0 = RIM_X - LANE_W/2.0
LANE_X1 = RIM_X + LANE_W/2.0

# Free-throw
FT_CY   = 19.0
FT_R    = 6.0

# Three-point
THREE_R        = 22.15
SIDELINE_INSET = 3.0
LEFT_POST_X    = SIDELINE_INSET
RIGHT_POST_X   = COURT_W - SIDELINE_INSET

# Center circle at half court
CENTER_C_R = 6.0

def court_shapes():
    lw = 2
    shapes = []

    def line(x0,y0,x1,y1, dash=None):
        return dict(type="line", xref="x", yref="y",
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(width=lw, color="black",
                              dash=("solid" if not dash else dash)))

    def rect(x0,y0,x1,y1, fill="rgba(0,0,0,0)"):
        return dict(type="rect", xref="x", yref="y",
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(width=lw, color="black"), fillcolor=fill)

    def circle(cx,cy,r, dash=None):
        return dict(type="circle", xref="x", yref="y",
                    x0=cx-r, y0=cy-r, x1=cx+r, y1=cy+r,
                    line=dict(width=lw, color="black",
                              dash=("solid" if not dash else dash)),
                    fillcolor="rgba(0,0,0,0)")

    # Outer border
    shapes.append(rect(0, 0, COURT_W, HALF_H))

    # Lane (paint) + free-throw line
    shapes.append(rect(LANE_X0, 0, LANE_X1, FT_CY))
    shapes.append(line(LANE_X0, FT_CY, LANE_X1, FT_CY))

    # Free-throw circle (full) + dashed bottom-facing semicircle
    shapes.append(circle(RIM_X, FT_CY, FT_R))
    t_vals = np.linspace(0, np.pi, 128)
    ft_path = " ".join(
        [f"M {RIM_X + FT_R*np.cos(t_vals[0])} {FT_CY + FT_R*np.sin(t_vals[0])}"] +
        [f"L {RIM_X + FT_R*np.cos(t)} {FT_CY + FT_R*np.sin(t)}" for t in t_vals[1:]]
    )
    shapes.append(dict(type="path", xref="x", yref="y",
                       path=ft_path, line=dict(width=lw, color="black", dash="dash")))

    # Backboard + rim
    shapes.append(line(RIM_X - 3.0, BACKBOARD_Y, RIM_X + 3.0, BACKBOARD_Y))
    shapes.append(circle(RIM_X, RIM_Y, RIM_R))

    # Tiny connector from backboard to the BACK of the rim (no line through rim)
    # Back of rim (closest to board) is at y = RIM_Y - RIM_R
    shapes.append(line(RIM_X, BACKBOARD_Y, RIM_X, RIM_Y - RIM_R))

    # Restricted-area semicircle (opening upward)
    t_vals_ra = np.linspace(0, np.pi, 128)
    ra_path = " ".join(
        [f"M {RIM_X + RESTRICTED_R*np.cos(t_vals_ra[0])} {RIM_Y + RESTRICTED_R*np.sin(t_vals_ra[0])}"] +
        [f"L {RIM_X + RESTRICTED_R*np.cos(t)} {RIM_Y + RESTRICTED_R*np.sin(t)}" for t in t_vals_ra[1:]]
    )
    shapes.append(dict(type="path", xref="x", yref="y",
                       path=ra_path, line=dict(width=lw, color="black")))

    # Corner posts and 3PT arc
    def t_for_x(x_target):
        # x = RIM_X + THREE_R * sin(t)
        val = (x_target - RIM_X) / THREE_R
        val = max(-1.0, min(1.0, val))
        return math.asin(val)

    tL = t_for_x(LEFT_POST_X)
    tR = t_for_x(RIGHT_POST_X)
    yL = RIM_Y + THREE_R * math.cos(tL)
    yR = RIM_Y + THREE_R * math.cos(tR)

    shapes.append(line(LEFT_POST_X, 0, LEFT_POST_X, yL))
    shapes.append(line(RIGHT_POST_X, 0, RIGHT_POST_X, yR))

    ts = np.linspace(tL, tR, 512)
    pts = [(RIM_X + THREE_R*np.sin(t), RIM_Y + THREE_R*np.cos(t)) for t in ts]
    arc_path = " ".join([f"M {pts[0][0]} {pts[0][1]}"] + [f"L {x} {y}" for x, y in pts[1:]])
    shapes.append(dict(type="path", xref="x", yref="y",
                       path=arc_path, line=dict(width=lw, color="black")))

    # Half-court line + DOWNWARD half-court semicircle
    shapes.append(line(0, HALF_H, COURT_W, HALF_H))
    t_vals_cc = np.linspace(0, np.pi, 128)
    pts_cc = [(RIM_X + CENTER_C_R*np.cos(t), HALF_H - CENTER_C_R*np.sin(t)) for t in t_vals_cc]
    cc_path = " ".join([f"M {pts_cc[0][0]} {pts_cc[0][1]}"] + [f"L {x} {y}" for x, y in pts_cc[1:]])
    shapes.append(dict(type="path", xref="x", yref="y",
                       path=cc_path, line=dict(width=lw, color="black")))

    return shapes

def base_fig():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=520, height=900,
        xaxis=dict(
            range=[0, COURT_W],
            showgrid=False, zeroline=False,
            ticks="", showticklabels=False,   # hide x-axis completely
            mirror=True,
            fixedrange=True
        ),
        yaxis=dict(
            range=[0, 48],                    # hard cap just above half court
            showgrid=False, zeroline=False,
            ticks="", showticklabels=False,   # hide y-axis completely
            scaleanchor="x", scaleratio=1,
            mirror=True,
            fixedrange=True
        ),
        shapes=court_shapes(),
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select",
        dragmode=False
    )
    return fig

# ---------------- Dash app ----------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Shot Chart — click the court to get (x, y)", style={"textAlign": "center"}),
    html.Div(
        dcc.Graph(
            id="court",
            figure=base_fig(),
            clear_on_unhover=True,
            config={"displayModeBar": True},
            style={"width": "540px"}  # fixed width → easy centering
        ),
        style={"display": "flex", "justifyContent": "center"}
    ),
    html.Div(id="coord_readout",
             style={"marginTop": "8px", "fontStyle": "italic", "textAlign": "center"})
])

@app.callback(
    Output("coord_readout", "children"),
    Input("court", "clickData"),
    prevent_initial_call=True
)
def show_click(clickData):
    if not clickData:
        return no_update
    x = clickData["points"][0]["x"]
    y = clickData["points"][0]["y"]
    return f"Selected: x={x:.2f}, y={y:.2f} ft"

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
