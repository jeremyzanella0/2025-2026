#Mark 2
#Adds ability to click and enter a possession (limited)

import math
import numpy as np
import dash
from dash import html, dcc, Output, Input, State, no_update, ctx
import plotly.graph_objects as go

# parser must be in the SAME folder
try:
    import pscoding18
except ImportError:
    pscoding18 = None
    print("Warning: pscoding18 not found - using dummy parser")

# ---------------- Court geometry (feet) ----------------
COURT_W = 50.0
HALF_H  = 47.0

RIM_X   = 25.0
RIM_Y   = 5.25
RIM_R   = 0.75
BACKBOARD_Y = 4.0
RESTRICTED_R = 4.0

LANE_W  = 16.0
LANE_X0 = RIM_X - LANE_W/2.0
LANE_X1 = RIM_X + LANE_W/2.0

FT_CY   = 19.0
FT_R    = 6.0

THREE_R        = 22.15
SIDELINE_INSET = 3.0
LEFT_POST_X    = SIDELINE_INSET
RIGHT_POST_X   = COURT_W - SIDELINE_INSET

CENTER_C_R = 6.0

def court_lines():
    """Generate court lines as trace data"""
    traces = []
    
    # Court boundary
    boundary_x = [0, COURT_W, COURT_W, 0, 0]
    boundary_y = [0, 0, HALF_H, HALF_H, 0]
    
    traces.append(go.Scatter(
        x=boundary_x, y=boundary_y,
        mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='boundary'
    ))
    
    # Lane rectangle
    lane_x = [LANE_X0, LANE_X1, LANE_X1, LANE_X0, LANE_X0]
    lane_y = [0, 0, FT_CY, FT_CY, 0]
    
    traces.append(go.Scatter(
        x=lane_x, y=lane_y,
        mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='lane'
    ))
    
    # Free throw circle
    theta = np.linspace(0, 2*np.pi, 100)
    ft_circle_x = RIM_X + FT_R * np.cos(theta)
    ft_circle_y = FT_CY + FT_R * np.sin(theta)
    
    traces.append(go.Scatter(
        x=ft_circle_x, y=ft_circle_y,
        mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='ft_circle'
    ))
    
    # 3-point arc
    def t_for_x(x_target):
        val = (x_target - RIM_X) / THREE_R
        val = max(-1.0, min(1.0, val))
        return math.asin(val)

    tL = t_for_x(LEFT_POST_X)
    tR = t_for_x(RIGHT_POST_X)
    yL = RIM_Y + THREE_R * math.cos(tL)
    yR = RIM_Y + THREE_R * math.cos(tR)
    
    # Left corner
    traces.append(go.Scatter(
        x=[LEFT_POST_X, LEFT_POST_X], y=[0, yL],
        mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='left_corner'
    ))
    
    # Right corner  
    traces.append(go.Scatter(
        x=[RIGHT_POST_X, RIGHT_POST_X], y=[0, yR],
        mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='right_corner'
    ))
    
    # 3-point arc
    ts = np.linspace(tL, tR, 100)
    arc_x = RIM_X + THREE_R * np.sin(ts)
    arc_y = RIM_Y + THREE_R * np.cos(ts)
    
    traces.append(go.Scatter(
        x=arc_x, y=arc_y,
        mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='three_arc'
    ))
    
    # Rim
    rim_theta = np.linspace(0, 2*np.pi, 50)
    rim_x = RIM_X + RIM_R * np.cos(rim_theta)
    rim_y = RIM_Y + RIM_R * np.sin(rim_theta)
    
    traces.append(go.Scatter(
        x=rim_x, y=rim_y,
        mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='rim'
    ))
    
    # Backboard
    traces.append(go.Scatter(
        x=[RIM_X - 3.0, RIM_X + 3.0], y=[BACKBOARD_Y, BACKBOARD_Y],
        mode='lines',
        line=dict(width=3, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='backboard'
    ))
    
    # Restricted area (semicircle)
    ra_theta = np.linspace(0, np.pi, 50)
    ra_x = RIM_X + RESTRICTED_R * np.cos(ra_theta)
    ra_y = RIM_Y + RESTRICTED_R * np.sin(ra_theta)
    
    traces.append(go.Scatter(
        x=ra_x, y=ra_y,
        mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='restricted'
    ))
    
    # Half court line
    traces.append(go.Scatter(
        x=[0, COURT_W], y=[HALF_H, HALF_H],
        mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False,
        hoverinfo='skip',
        name='halfcourt'
    ))
    
    return traces

def make_click_dots():
    """Create small visible dots that look like court markings but are clickable"""
    
    # Create a strategic grid of click points
    xs = []
    ys = []
    
    # More dense grid for better coverage
    for x in range(3, int(COURT_W)-2, 4):  # Every 4 feet from x=3 to x=47
        for y in range(3, int(HALF_H)-2, 4):  # Every 4 feet from y=3 to y=43
            xs.append(x)
            ys.append(y)
    
    return go.Scatter(
        x=xs, y=ys,
        mode='markers',
        marker=dict(
            size=6,
            color='lightblue',
            opacity=0.4,
            line=dict(width=1, color='darkblue')
        ),
        showlegend=False,
        hovertemplate='<b>Click here</b><br>x=%{x}<br>y=%{y}<extra></extra>',
        name='click_zones'
    )

def base_fig():
    fig = go.Figure()
    
    # Add court lines first
    for trace in court_lines():
        fig.add_trace(trace)
    
    # Add visible click dots on top
    fig.add_trace(make_click_dots())
    
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=520, height=900,
        xaxis=dict(
            range=[0, COURT_W], 
            showgrid=False, 
            zeroline=False,
            ticks="", 
            showticklabels=False, 
            mirror=True, 
            fixedrange=True
        ),
        yaxis=dict(
            range=[0, HALF_H], 
            showgrid=False, 
            zeroline=False,
            ticks="", 
            showticklabels=False, 
            scaleanchor="x", 
            scaleratio=1,
            mirror=True, 
            fixedrange=True
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select",
        dragmode=False
    )
    return fig

# ---------------- Dash app ----------------
app = dash.Dash(__name__)
app.title = "Shot Chart — click the blue dots to enter a possession"

def modal_style(show: bool):
    return {
        "display": "flex" if show else "none",
        "position": "fixed",
        "top": 0, "left": 0, "right": 0, "bottom": 0,
        "backgroundColor": "rgba(0,0,0,0.35)",
        "alignItems": "center",
        "justifyContent": "center",
        "zIndex": 1000,
    }

app.layout = html.Div([
    html.H2("Shot Chart — click the blue dots to enter a possession", style={"textAlign": "center"}),
    html.Div("(Blue dots show clickable areas on the court)", style={"textAlign": "center", "color": "#666", "fontSize": "14px"}),

    html.Div(
        dcc.Graph(
            id="court",
            figure=base_fig(),
            config={"displayModeBar": False},
            style={"width": "540px"}
        ),
        style={"display": "flex", "justifyContent": "center"}
    ),

    html.Div(id="debug_click", style={"fontSize": "12px", "textAlign": "center", "color": "#666"}),

    # Modal
    html.Div(
        id="input_modal",
        style=modal_style(False),
        children=html.Div(
            style={
                "width": "520px",
                "background": "white",
                "borderRadius": "12px",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.2)",
                "padding": "18px"
            },
            children=[
                html.H3("Enter possession shorthand"),
                html.Div(id="click_xy_label", style={"color": "#666", "marginBottom": "6px"}),
                dcc.Input(
                    id="possession_input",
                    type="text",
                    placeholder="e.g. 1/2pnr3/45chd+",
                    style={"width": "100%", "padding": "10px"},
                    debounce=False
                ),
                html.Div(
                    style={"display": "flex", "gap": "8px", "marginTop": "12px", "justifyContent": "flex-end"},
                    children=[
                        html.Button("Cancel", id="btn_cancel", n_clicks=0,
                                    style={"padding": "8px 14px", "borderRadius": "8px", "border": "1px solid #ccc", "background": "white"}),
                        html.Button("Submit", id="btn_submit", n_clicks=0,
                                    style={"padding": "8px 14px", "borderRadius": "8px", "border": "none", "background": "#2563eb", "color": "white"})
                    ]
                )
            ]
        )
    ),

    html.Div(id="output_block",
             style={"maxWidth": "680px", "margin": "18px auto", "padding": "0 12px"}),

    dcc.Store(id="store_modal_open", data=False),
    dcc.Store(id="store_last_click_xy", data=None),
])

# ========= Callbacks =========

# Debug callback
@app.callback(
    Output("debug_click", "children"),
    Input("court", "clickData"),
    prevent_initial_call=True
)
def debug_click_data(clickData):
    if clickData is None:
        return "No clicks detected"
    return f"Last click: x={clickData['points'][0]['x']:.1f}, y={clickData['points'][0]['y']:.1f}"

# Main controller
@app.callback(
    Output("store_modal_open", "data"),
    Output("store_last_click_xy", "data"),
    Output("click_xy_label", "children"),
    Input("court", "clickData"),
    Input("btn_cancel", "n_clicks"),
    Input("btn_submit", "n_clicks"),
    Input("possession_input", "n_submit"),
    prevent_initial_call=True
)
def control_modal(clickData, cancel_clicks, submit_clicks, n_submit):
    if not ctx.triggered:
        return no_update, no_update, no_update
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Open on court click
    if trigger_id == "court" and clickData and clickData.get("points"):
        x = float(clickData["points"][0]["x"])
        y = float(clickData["points"][0]["y"])
        return True, {"x": x, "y": y}, f"Selected: x={x:.2f}, y={y:.2f} ft"

    # Close on cancel OR submit
    if trigger_id in ("btn_cancel", "btn_submit", "possession_input"):
        return False, no_update, no_update

    return no_update, no_update, no_update

# Modal visibility
@app.callback(
    Output("input_modal", "style"),
    Input("store_modal_open", "data")
)
def toggle_modal(open_flag):
    return modal_style(bool(open_flag))

# Parse possession
@app.callback(
    Output("output_block", "children"),
    Input("btn_submit", "n_clicks"),
    Input("possession_input", "n_submit"),
    State("possession_input", "value"),
    State("store_last_click_xy", "data"),
    prevent_initial_call=True
)
def parse_and_render(btn_clicks, n_submit, possession_text, xy):
    triggered = (btn_clicks or 0) > 0 or (n_submit or 0) > 0
    if not triggered:
        return no_update

    if not possession_text or not possession_text.strip():
        return html.Div("Please enter a possession string.", style={"color": "crimson"})

    try:
        if pscoding18:
            parsed = pscoding18.parse_possession_string(possession_text.strip())
            lines = [parsed] if isinstance(parsed, str) else list(parsed)
        else:
            lines = [f"Dummy parse: {possession_text.strip()}"]
    except Exception as e:
        return html.Div(f"Error while parsing: {e}", style={"color": "crimson"})

    xy_line = f"Location: x={xy['x']:.2f}, y={xy['y']:.2f} ft\n" if xy else ""
    return html.Div([
        html.H4("Play-by-Play"),
        html.Pre(xy_line + "\n".join(lines), style={
            "background": "#0b1021",
            "color": "#e6edf3",
            "padding": "12px",
            "borderRadius": "10px",
            "whiteSpace": "pre-wrap"
        })
    ])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)