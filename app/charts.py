"""Plotly figures for the dashboard. Each function takes tidy data and returns a figure."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import MODEL_COLORS, MODEL_LABELS

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"
BETTER = "#2a78d6"
WORSE = "#e34948"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _style(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family=FONT),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def history_chart(panel, overlay=None, overlay_label=None):
    """NY unemployment since 1976. An overlay indicator goes in its own panel below (never a second y-axis)."""
    rows = 2 if overlay else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.6, 0.4] if overlay else [1.0])
    fig.add_trace(go.Scatter(x=panel["date"], y=panel["ny_ur"], name="NY unemployment rate",
                             line=dict(color=INK, width=2),
                             hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_vrect(x0="2020-03-01", x1="2021-12-01", fillcolor="#f0efec", line_width=0, layer="below")
    fig.add_annotation(x="2020-05-01", y=16.7, text="COVID peak 16.7%", showarrow=False,
                       xanchor="right", xshift=-8, font=dict(color=INK, size=12))
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    if overlay:
        fig.add_trace(go.Scatter(x=panel["date"], y=panel[overlay], name=overlay_label,
                                 line=dict(color=MODEL_COLORS["combo_s"], width=2),
                                 hovertemplate="%{x|%b %Y}: %{y:,.1f}<extra></extra>"), row=2, col=1)
        fig.update_yaxes(title_text=overlay_label, title_font=dict(size=12), row=2, col=1)
    fig.update_layout(hovermode="x unified", showlegend=bool(overlay))
    return _style(fig, 460 if overlay else 340)


def explorer_chart(forecasts, panel, models, horizon, window):
    """Actual NYUR against each model's forecasts made `horizon` months earlier, plus live forecasts."""
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    fig = go.Figure()

    actual = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    fig.add_trace(go.Scatter(x=actual["date"], y=actual["ny_ur"], name="Actual",
                             line=dict(color=INK, width=3),
                             hovertemplate="Actual: %{y:.1f}%<extra></extra>"))

    shown = forecasts[(forecasts["horizon"] == horizon) & forecasts["model"].isin(models)]
    for model in models:
        rows = shown[(shown["model"] == model) & (shown["period"] != "live")
                     & (shown["date"] >= start) & (shown["date"] <= end)].sort_values("date")
        if rows.empty:
            continue
        dash = "dot" if model == "naive" else "solid"
        fig.add_trace(go.Scatter(x=rows["date"], y=rows["yhat"], name=MODEL_LABELS[model],
                                 line=dict(color=MODEL_COLORS[model], width=2, dash=dash),
                                 hovertemplate=MODEL_LABELS[model] + ": %{y:.2f}%<extra></extra>"))

    if start < pd.Timestamp("2022-01-01") < end:
        fig.add_vline(x="2022-01-01", line=dict(color="#c3c9d0", width=1, dash="dot"))
        fig.add_annotation(x="2022-01-01", y=1, yref="paper", text="Test period starts →", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=INK_SECONDARY, size=12))
    if start < pd.Timestamp("2025-10-01") < end:
        fig.add_annotation(x="2025-10-01", y=0, yref="paper", text="Oct ’25 gap (shutdown)", showarrow=False,
                           yanchor="bottom", font=dict(color=INK_SECONDARY, size=11))

    # Live forecasts only make sense when the window reaches the present.
    if end >= pd.Timestamp("2026-08-01"):
        live = forecasts[(forecasts["period"] == "live") & forecasts["model"].isin(models)]
        live = live[live["date"] <= pd.Timestamp("2027-03-01")]
        # Combo 95% range from the latest forecast date, anchored at the last published actual.
        band = live[(live["model"] == "combo_s") & (live["origin"] == live["origin"].max())].sort_values("date")
        if not band.empty:
            last = actual.dropna(subset=["ny_ur"]).iloc[-1]
            x = [last["date"]] + band["date"].tolist()
            fig.add_trace(go.Scatter(x=x + x[::-1],
                                     y=[last["ny_ur"]] + band["upper"].tolist()
                                       + band["lower"].tolist()[::-1] + [last["ny_ur"]],
                                     mode="lines", fill="toself", fillcolor="rgba(42,120,214,0.16)", line=dict(width=0),
                                     name="Combo 95% range", hoverinfo="skip"))
        for model in models:
            rows = live[live["model"] == model].sort_values("date")
            if rows.empty:
                continue
            fig.add_trace(go.Scatter(x=rows["date"], y=rows["yhat"], mode="markers",
                                     name=MODEL_LABELS[model] + " · live", showlegend=False,
                                     marker=dict(color=MODEL_COLORS[model], size=9,
                                                 line=dict(color=SURFACE, width=2)),
                                     hovertemplate=MODEL_LABELS[model] + " live: %{y:.2f}%<extra></extra>"))
        fig.add_vrect(x0="2026-08-16", x1="2027-03-15", fillcolor="#eef3fa", line_width=0, layer="below",
                      annotation_text="Live", annotation_position="top left",
                      annotation_font=dict(color=MODEL_COLORS["combo_s"], size=12))

    # Fit the y-axis to actuals and point forecasts; the wide 6-month band is clipped, as in the mockup.
    values = pd.concat([actual["ny_ur"], shown.loc[(shown["date"] >= start) & (shown["date"] <= end), "yhat"]]).dropna()
    fig.update_yaxes(ticksuffix="%", range=[values.min() - 0.4, values.max() + 0.4])
    fig.update_layout(hovermode="x unified")
    return _style(fig, 420)


def scorecard_chart(metrics, view, horizon):
    """Horizontal bars of RMSE / naive for one period and horizon; the line at 1.0 is naive."""
    rows = metrics[(metrics["view"] == view) & (metrics["horizon"] == horizon)
                   & (metrics["model"] != "naive")].sort_values("rmse_vs_naive", ascending=False)
    labels = [MODEL_LABELS[m] for m in rows["model"]]
    colors = [BETTER if r < 1 else WORSE for r in rows["rmse_vs_naive"]]
    fig = go.Figure(go.Bar(
        x=rows["rmse_vs_naive"], y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[f"{r:.2f}×" for r in rows["rmse_vs_naive"]], textposition="outside",
        textfont=dict(color=INK_SECONDARY),
        customdata=rows[["rmse", "n"]], cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.2f}× naive error<br>RMSE %{customdata[0]:.3f} pp · %{customdata[1]} months<extra></extra>",
    ))
    fig.add_vline(x=1, line=dict(color=INK, width=2))
    fig.add_annotation(x=1, y=1.02, yref="paper", text="naive", showarrow=False,
                       yanchor="bottom", font=dict(color=INK, size=12))
    top = max(1.3, rows["rmse_vs_naive"].max() * 1.15)
    fig.update_xaxes(range=[0, top], showgrid=True, gridcolor=GRID, title_text="Forecast error ÷ naive's error")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(showlegend=False, bargap=0.35)
    return _style(fig, 320)


def ablation_chart(ablation, period):
    """How much each Ridge variant beats a Ridge that only knows the unemployment rate's own history."""
    rows = ablation[ablation["period"] == period]
    names = {"ridge": "Core indicators (claims, payrolls, US rate)",
             "ridge_regional": "Core + regional (NJ/CT/PA, Empire survey)"}
    fig = go.Figure()
    for model, color in [("ridge", MODEL_COLORS["combo_s"]), ("ridge_regional", MODEL_COLORS["var_s"])]:
        r = rows[rows["model"] == model]
        fig.add_trace(go.Bar(x=[f"{h} mo" for h in r["horizon"]], y=r["improvement_pct"], name=names[model],
                             marker=dict(color=color, cornerradius=4),
                             text=[f"{v:+.0f}%" for v in r["improvement_pct"]], textposition="outside",
                             cliponaxis=False,
                             hovertemplate=names[model] + "<br>%{x}: %{y:+.1f}% lower error<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    fig.update_yaxes(ticksuffix="%", title_text="Error reduction vs. own-history only")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    return _style(fig, 340)


def sparkline(dates, values, color=INK):
    """A small trend line for an indicator tile: no axes, hover shows the value."""
    fig = go.Figure(go.Scatter(x=dates, y=values, mode="lines", line=dict(color=color, width=2),
                               hovertemplate="%{x|%b %Y}: %{y:,.1f}<extra></extra>"))
    fig.update_layout(height=90, margin=dict(l=0, r=0, t=4, b=0), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      showlegend=False, hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def paired_chart(panel, column, label, start):
    """NY unemployment on top and one indicator below, sharing the time axis (no second y-axis)."""
    rows = panel[panel["date"] >= start]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=rows["date"], y=rows["ny_ur"], name="NY unemployment rate",
                             line=dict(color=INK, width=2), hovertemplate="%{y:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rows["date"], y=rows[column], name=label,
                             line=dict(color=MODEL_COLORS["combo_s"], width=2),
                             hovertemplate="%{y:,.1f}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_layout(hovermode="x unified")
    return _style(fig, 380)
