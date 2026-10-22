"""KKD Real Estate Price Prediction Dashboard.

Pure UI Dashboard:
- Collects chosen house parameters from interactive input frames
- "Look for a house" button triggers prediction
- Passes chosen parameters directly into `predict_price_model(params)`
- Displays the predicted price and house summary
"""
import os
import numpy as np
from dash import Dash, html, dcc, Input, Output, State

from kkd_data import (
    load_dataset,
    encode_house,
    DISTRICT_IDS,
    DISTRICTS,
    SCHOOLS,
    AGENTS,
    COLORS,
    ADVERTISEMENTS,
    REF_YEAR
)

# Load dataset stats (kept in backend for the model inference function)
_, _, STATS = load_dataset()

# Model file path
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.npz")


def get_trained_model():
    """Loads model.npz if your teammate has saved it."""
    if os.path.exists(MODEL_FILE):
        try:
            data = np.load(MODEL_FILE, allow_pickle=True)
            w = data["w"]
            b = float(data["b"])
            scale_type = str(data["scale_type"]) if "scale_type" in data else "million"
            y_scale = float(data["y_scale"]) if "y_scale" in data else 1e6
            r2 = float(data["r2"]) if "r2" in data else None
            mae = float(data["mae"]) if "mae" in data else None
            return {"w": w, "b": b, "scale_type": scale_type, "y_scale": y_scale, "r2": r2, "mae": mae}
        except Exception as e:
            print("Warning: could not load model.npz:", e)
    return None


def predict_price_model(params):
    """
    >>> YOUR TEAMMATE PLUGS THEIR MODEL HERE <<<
    
    Receives chosen parameters as a dictionary:
      params = {
          'district_id': str,
          'year': int,
          'parking': str ('yes'/'no'),
          'remodeled': int (-1 or year),
          'size': float,
          'lot_w': float,
          'bathrooms': int,
          'kitchens': int,
          'color': str,
          'fireplace': str ('yes'/'no'),
          'advertisement': str ('no'/'regular'/'premium'),
          'condition_rating': int,
          'storage_rating': int,
          'sun_factor': float,
          ...
      }
    """
    model = get_trained_model()
    if model is not None:
        # Preprocessing & normalization happens here inside the model logic
        x_norm = encode_house(params, STATS)
        pred_scaled = float(np.dot(x_norm, model["w"]) + model["b"])

        if model["scale_type"] == "log":
            price_nok = np.exp(pred_scaled)
        else:
            price_nok = pred_scaled * model["y_scale"]

        price_text = f"{max(0.0, price_nok):,.0f} NOK"
        status_badge = "🟢 Model Active"
        r2_str = f"R² = {model['r2']:.3f}" if model["r2"] is not None else ""
        mae_str = f"MAE = {model['mae']:,.0f} NOK" if model["mae"] is not None else ""
        metrics_text = f"Linear Model: {r2_str} | {mae_str}" if (r2_str or mae_str) else "Computed from model.npz"
        return price_text, status_badge, metrics_text

    # Default message while teammate finishes the model
    return (
        "Awaiting model.npz",
        "⏳ Teammate Model Pending",
        "Parameters captured successfully and ready for model inference."
    )


# District dropdown options
district_options = [
    {
        "label": f"District {i} (Crime: {DISTRICTS[d_id]['crime_rating']}/5, Transport: {DISTRICTS[d_id]['public_transport_rating']}/5)"
                 + (" [Reference]" if i == 0 else ""),
        "value": d_id
    }
    for i, d_id in enumerate(DISTRICT_IDS)
]

# Styling helpers
CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "10px",
    "padding": "22px",
    "marginBottom": "20px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
    "border": "1px solid #e9ecef"
}

LABEL_STYLE = {
    "fontWeight": "600",
    "fontSize": "13px",
    "color": "#495057",
    "marginBottom": "6px",
    "display": "block"
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "8px 10px",
    "borderRadius": "6px",
    "border": "1px solid #ced4da",
    "fontSize": "14px",
    "boxSizing": "border-box"
}

GRID_2COL = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr",
    "gap": "15px",
    "marginBottom": "15px"
}

GRID_3COL = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr 1fr",
    "gap": "15px",
    "marginBottom": "15px"
}

BUTTON_STYLE = {
    "width": "100%",
    "backgroundColor": "#1e3a8a",
    "color": "#ffffff",
    "padding": "14px 20px",
    "fontSize": "16px",
    "fontWeight": "700",
    "borderRadius": "8px",
    "border": "none",
    "cursor": "pointer",
    "boxShadow": "0 4px 10px rgba(30, 58, 138, 0.25)",
    "letterSpacing": "0.5px"
}

# Initialize Dash App
app = Dash(__name__)
app.title = "KKD - Real Estate Dashboard"

app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif",
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh",
        "padding": "30px 40px",
        "color": "#212529"
    },
    children=[
        # Header
        html.Div(
            style={"marginBottom": "25px", "borderBottom": "2px solid #dee2e6", "paddingBottom": "15px"},
            children=[
                html.H1(
                    "Knut-Knut House Development (KKD)",
                    style={"margin": "0 0 5px 0", "color": "#1e3a8a", "fontSize": "28px"}
                ),
                html.P(
                    "Select house parameters and click 'Look for a house' to get the predicted valuation.",
                    style={"margin": "0", "color": "#6c757d", "fontSize": "15px"}
                )
            ]
        ),

        # Main Layout: 2 Columns
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.2fr 1fr",
                "gap": "30px",
                "alignItems": "start"
            },
            children=[
                # LEFT COLUMN: INPUT FRAMES + LOOK FOR A HOUSE BUTTON
                html.Div([
                    # Section 1: Location & School
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3("📍 1. Location & School", style={"marginTop": "0", "fontSize": "18px", "color": "#1e3a8a"}),
                            html.Div(
                                style={"marginBottom": "12px"},
                                children=[
                                    html.Label("District:", style=LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="input-district",
                                        options=district_options,
                                        value=DISTRICT_IDS[0],
                                        clearable=False
                                    )
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Local School (Rating & Capacity):", style=LABEL_STYLE),
                                    dcc.Dropdown(id="input-school", clearable=False)
                                ]
                            )
                        ]
                    ),

                    # Section 2: Property Dimensions & Age
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3("🏠 2. Property Dimensions & Age", style={"marginTop": "0", "fontSize": "18px", "color": "#1e3a8a"}),
                            html.Div(
                                style=GRID_2COL,
                                children=[
                                    html.Div([
                                        html.Label("Construction Year:", style=LABEL_STYLE),
                                        dcc.Input(id="input-year", type="number", value=1990, min=1900, max=REF_YEAR, style=INPUT_STYLE)
                                    ]),
                                    html.Div([
                                        html.Label("Living Size (m²):", style=LABEL_STYLE),
                                        dcc.Input(id="input-size", type="number", value=95, min=15, max=600, style=INPUT_STYLE)
                                    ])
                                ]
                            ),
                            html.Div(
                                style=GRID_3COL,
                                children=[
                                    html.Div([
                                        html.Label("Lot Width (m):", style=LABEL_STYLE),
                                        dcc.Input(id="input-lot-w", type="number", value=35, min=0, max=200, style=INPUT_STYLE)
                                    ]),
                                    html.Div([
                                        html.Label("Bathrooms:", style=LABEL_STYLE),
                                        dcc.Input(id="input-bathrooms", type="number", value=2, min=1, max=10, style=INPUT_STYLE)
                                    ]),
                                    html.Div([
                                        html.Label("Kitchens:", style=LABEL_STYLE),
                                        dcc.Input(id="input-kitchens", type="number", value=1, min=1, max=5, style=INPUT_STYLE)
                                    ])
                                ]
                            )
                        ]
                    ),

                    # Section 3: Renovation & Amenities
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3("🛠️ 3. Renovation & Amenities", style={"marginTop": "0", "fontSize": "18px", "color": "#1e3a8a"}),
                            html.Div(
                                style=GRID_2COL,
                                children=[
                                    html.Div([
                                        html.Label("Was Remodeled?", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="input-was-remodeled",
                                            options=[{"label": "Yes", "value": "yes"}, {"label": "Never (-1)", "value": "no"}],
                                            value="yes",
                                            clearable=False
                                        )
                                    ]),
                                    html.Div([
                                        html.Label("Remodel Year:", style=LABEL_STYLE),
                                        dcc.Input(id="input-remodeled-year", type="number", value=2015, min=1950, max=REF_YEAR, style=INPUT_STYLE)
                                    ])
                                ]
                            ),
                            html.Div(
                                style=GRID_3COL,
                                children=[
                                    html.Div([
                                        html.Label("Parking:", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="input-parking",
                                            options=[{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}],
                                            value="yes",
                                            clearable=False
                                        )
                                    ]),
                                    html.Div([
                                        html.Label("Fireplace:", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="input-fireplace",
                                            options=[{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}],
                                            value="no",
                                            clearable=False
                                        )
                                    ]),
                                    html.Div([
                                        html.Label("External Storage (m²):", style=LABEL_STYLE),
                                        dcc.Input(id="input-ext-storage", type="number", value=0, min=0, max=100, style=INPUT_STYLE)
                                    ])
                                ]
                            )
                        ]
                    ),

                    # Section 4: Condition & Marketing
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3("🎨 4. Condition & Marketing", style={"marginTop": "0", "fontSize": "18px", "color": "#1e3a8a"}),
                            html.Div(
                                style=GRID_2COL,
                                children=[
                                    html.Div([
                                        html.Label("House Color:", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="input-color",
                                            options=[{"label": c.capitalize(), "value": c} for c in COLORS],
                                            value="white",
                                            clearable=False
                                        )
                                    ]),
                                    html.Div([
                                        html.Label("Advertisement Tier:", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="input-ad",
                                            options=[{"label": a.capitalize(), "value": a} for a in ADVERTISEMENTS],
                                            value="regular",
                                            clearable=False
                                        )
                                    ])
                                ]
                            ),
                            html.Div(
                                style={"marginBottom": "12px"},
                                children=[
                                    html.Div([
                                        html.Label(id="label-condition", children="Condition Rating: 6 / 10", style=LABEL_STYLE),
                                        dcc.Slider(id="input-condition", min=1, max=10, step=1, value=6, marks={i: str(i) for i in range(1, 11)})
                                    ])
                                ]
                            ),
                            html.Div(
                                style={"marginBottom": "12px"},
                                children=[
                                    html.Div([
                                        html.Label(id="label-storage", children="Storage Rating: 6 / 10", style=LABEL_STYLE),
                                        dcc.Slider(id="input-storage", min=1, max=10, step=1, value=6, marks={i: str(i) for i in range(1, 11)})
                                    ])
                                ]
                            ),
                            html.Div(
                                style={"marginBottom": "15px"},
                                children=[
                                    html.Div([
                                        html.Label(id="label-sun", children="Sun Factor: 0.60", style=LABEL_STYLE),
                                        dcc.Slider(id="input-sun", min=0.0, max=1.0, step=0.05, value=0.60, marks={0: "0.0", 0.5: "0.5", 1.0: "1.0"})
                                    ])
                                ]
                            )
                        ]
                    ),

                    # Action Button
                    html.Div(
                        style={"marginBottom": "20px"},
                        children=[
                            html.Button(
                                "🔍 Look for a house",
                                id="btn-look-house",
                                n_clicks=0,
                                style=BUTTON_STYLE
                            )
                        ]
                    )
                ]),

                # RIGHT COLUMN: PREDICTION & SELECTED HOUSE SUMMARY
                html.Div([
                    # Main Price Prediction Card
                    html.Div(
                        style={
                            **CARD_STYLE,
                            "background": "linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%)",
                            "color": "#ffffff",
                            "border": "none",
                            "padding": "30px"
                        },
                        children=[
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                children=[
                                    html.H4("PREDICTED VALUATION", style={"margin": "0", "fontSize": "13px", "letterSpacing": "1px", "opacity": "0.9"}),
                                    html.Span(id="model-status-badge", style={"fontSize": "12px", "padding": "4px 12px", "borderRadius": "12px", "backgroundColor": "rgba(255,255,255,0.2)"})
                                ]
                            ),
                            html.Div(
                                id="predicted-price-display",
                                style={"fontSize": "42px", "fontWeight": "bold", "margin": "20px 0 10px 0"},
                                children="-- NOK"
                            ),
                            html.Div(
                                id="model-metrics-display",
                                style={"fontSize": "14px", "opacity": "0.9"},
                                children="Click 'Look for a house' to run valuation"
                            )
                        ]
                    ),

                    # Selected House Summary Card
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3("📋 Selected Property Overview", style={"marginTop": "0", "fontSize": "18px", "color": "#1e3a8a"}),
                            html.P("Current parameters sent to the valuation model:", style={"fontSize": "13px", "color": "#6c757d", "marginTop": "-5px"}),
                            html.Div(
                                id="selected-summary-display",
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "12px",
                                    "paddingTop": "10px"
                                }
                            )
                        ]
                    )
                ])
            ]
        )
    ]
)


# Callback 1: Dynamic School options based on selected District
@app.callback(
    Output("input-school", "options"),
    Output("input-school", "value"),
    Input("input-district", "value")
)
def update_schools_for_district(selected_district):
    district_schools = [s for s in SCHOOLS.values() if s["district_id"] == selected_district]
    options = [
        {
            "label": f"School built {s['built_year']} (Rating: {s['rating']}/5, Capacity: {s['capacity']})",
            "value": s["id"]
        }
        for s in district_schools
    ]
    default_val = options[0]["value"] if options else None
    return options, default_val


# Callback 2: Disable / Enable Remodel Year input based on was_remodeled
@app.callback(
    Output("input-remodeled-year", "disabled"),
    Input("input-was-remodeled", "value")
)
def toggle_remodel_input(was_remodeled):
    return was_remodeled == "no"


# Callback 3: Update Slider Text Labels
@app.callback(
    Output("label-condition", "children"),
    Output("label-storage", "children"),
    Output("label-sun", "children"),
    Input("input-condition", "value"),
    Input("input-storage", "value"),
    Input("input-sun", "value")
)
def update_slider_labels(cond, stor, sun):
    return (
        f"Condition Rating: {cond} / 10",
        f"Storage Rating: {stor} / 10",
        f"Sun Factor: {sun:.2f}"
    )


# Callback 4: Triggered by "Look for a house" button
@app.callback(
    Output("predicted-price-display", "children"),
    Output("model-status-badge", "children"),
    Output("model-metrics-display", "children"),
    Output("selected-summary-display", "children"),
    Input("btn-look-house", "n_clicks"),
    State("input-district", "value"),
    State("input-school", "value"),
    State("input-year", "value"),
    State("input-size", "value"),
    State("input-lot-w", "value"),
    State("input-bathrooms", "value"),
    State("input-kitchens", "value"),
    State("input-was-remodeled", "value"),
    State("input-remodeled-year", "value"),
    State("input-parking", "value"),
    State("input-fireplace", "value"),
    State("input-ext-storage", "value"),
    State("input-color", "value"),
    State("input-ad", "value"),
    State("input-condition", "value"),
    State("input-storage", "value"),
    State("input-sun", "value")
)
def compute_dashboard_state(
    n_clicks,
    district_id, school_id, year, size, lot_w, bathrooms, kitchens,
    was_remodeled, remodeled_year, parking, fireplace, ext_storage,
    color, ad_tier, condition, storage, sun
):
    # Guard against initial None inputs
    year = int(year or 1990)
    size = float(size or 95)
    lot_w = float(lot_w or 35)
    bathrooms = int(bathrooms or 2)
    kitchens = int(kitchens or 1)
    ext_storage = float(ext_storage or 0)
    remodeled_val = int(remodeled_year or 2015) if was_remodeled == "yes" else -1

    if not school_id:
        district_schools = [s["id"] for s in SCHOOLS.values() if s["district_id"] == district_id]
        school_id = district_schools[0] if district_schools else list(SCHOOLS.keys())[0]

    first_agent = list(AGENTS.keys())[0]

    # Clean dictionary with all chosen parameters
    chosen_params = {
        "year": year,
        "remodeled": remodeled_val,
        "size": size,
        "lot_w": lot_w,
        "bathrooms": bathrooms,
        "kitchens": kitchens,
        "condition_rating": condition,
        "storage_rating": storage,
        "external_storage_m2": ext_storage,
        "sun_factor": sun,
        "fireplace": fireplace,
        "parking": parking,
        "advertisement": ad_tier,
        "color": color,
        "district_id": district_id,
        "school_id": school_id,
        "agent_id": first_agent,
        "price": 0
    }

    # Pass directly to model function
    price_text, status_badge, metrics_text = predict_price_model(chosen_params)

    # Build friendly summary cards for right panel
    district_idx = DISTRICT_IDS.index(district_id) if district_id in DISTRICT_IDS else 0
    remodeled_text = "Never" if remodeled_val == -1 else f"In {remodeled_val}"

    summary_items = [
        html.Div([html.Span("District: ", style={"color": "#64748b", "fontSize": "13px"}), html.Strong(f"District {district_idx}")]),
        html.Div([html.Span("Year Built: ", style={"color": "#64748b", "fontSize": "13px"}), html.Strong(str(year))]),
        html.Div([html.Span("Living Size: ", style={"color": "#64748b", "fontSize": "13px"}), html.Strong(f"{size:.0f} m²")]),
        html.Div([html.Span("Parking: ", style={"color": "#64748b", "fontSize": "13px"}), html.Strong(parking.capitalize())]),
        html.Div([html.Span("Remodeled: ", style={"color": "#64748b", "fontSize": "13px"}), html.Strong(remodeled_text)]),
        html.Div([html.Span("Color: ", style={"color": "#64748b", "fontSize": "13px"}), html.Strong(color.capitalize())]),
        html.Div([html.Span("Bathrooms: ", style={"color": "#64748b", "fontSize": "13px"}), html.Strong(str(bathrooms))]),
        html.Div([html.Span("Condition: ", style={"color": "#64748b", "fontSize": "13px"}), html.Strong(f"{condition} / 10")]),
    ]

    return price_text, status_badge, metrics_text, summary_items


if __name__ == "__main__":
    app.run(debug=True)
