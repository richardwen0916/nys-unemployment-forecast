"""NY Jobs Outlook: forecasting dashboard for the New York State unemployment rate.

Run from the repo root:  streamlit run app/app.py
"""
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts
from assistant import get_api_key, stream_answer
from data import (MODEL_DESCRIPTIONS, MODEL_LABELS, MODELS, PERIOD_WINDOWS, PERIODS, load_ablation,
                  load_dictionary, load_forecasts, load_metrics, load_panel, load_taxable_sales)

st.set_page_config(page_title="NY Jobs Outlook", page_icon="📈", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>", unsafe_allow_html=True)

panel = load_panel()
forecasts = load_forecasts()
metrics = load_metrics()
latest = panel.dropna(subset=["ny_ur"]).iloc[-1]
live = forecasts[forecasts["period"] == "live"]

# Sample conversation shown in the assistant panel when no API key is set.
PREVIEW_CHAT = [
    ("user", "Why do the models expect unemployment to stay near 4.3%?"),
    ("ai", "The rate just fell from 4.6% to 4.3%, and NY claims and neighboring states aren’t signalling a new "
           "turn. In calm periods our recommended model leans almost entirely on the latest value, so it stays "
           "close to 4.3%. The 6-month range is wide (2.4–4.7%) because the models can’t foresee shocks."),
    ("user", "Do jobless claims actually help?"),
    ("ai", "At turning points, yes: in 2022 the claims-based ARX cut 3-month error by about a third. From 2023 on "
           "it did worse than “no change” at 3–6 months. See the Model scorecard tab."),
]
SUGGESTIONS = ["Explain pseudo-real-time", "Which model should I trust?", "What happened in COVID?"]

st.markdown(f"""
<div class="brand">
  <div class="brand-left">
    <svg width="30" height="30" viewBox="0 0 28 28" aria-hidden="true"><rect width="28" height="28" rx="6" fill="#15202B"></rect>
      <path d="M6 19 L11 13 L15 16 L22 8" stroke="#fff" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"></path></svg>
    <div><div class="brand-name">NY Jobs Outlook</div><div class="brand-sub">MGMTMSA 437 · Team project</div></div>
  </div>
  <div class="chip">Data as of 19 Oct 2026 · latest rate {latest['date']:%b %Y}</div>
</div>
""", unsafe_allow_html=True)

# The original overlay Indicators page is kept below but hidden. Set True to bring its tab back.
SHOW_OLD_INDICATORS = False
tab_names = ["Dashboard", "Model scorecard", "Indicators", "Methodology"]
tabs = st.tabs(tab_names + (["Indicators (old overlay)"] if SHOW_OLD_INDICATORS else []))
dashboard, scorecard, indicators_alt, method = tabs[:4]

# ---------------------------------------------------------------- Dashboard
with dashboard:
    combo_sep = live[(live["model"] == "combo_s") & (live["date"] == "2026-09-01")].iloc[0]
    prior = panel.dropna(subset=["ny_ur"]).iloc[-3]
    # Recommended-model tile: error vs. naive over the 9 test cases (3 test periods x 3 horizons).
    tests = metrics[metrics["view"].isin(["post_covid", "calm_2023on", "covid"])]
    summary = tests.groupby("model")["rmse_vs_naive"].agg(["min", "mean", "max"]) - 1
    combo, ml = summary.loc["combo_s"], summary.loc["ml"]
    # 441 = forecast origins Jan 1990-Sep 2026 in modeling/results/experiments/step3_ml/origin_metadata.csv

    st.markdown(f"""
<div class="eyebrow">Research question</div>
<div class="hero">Can economic indicators beat “no change” when forecasting New York’s unemployment rate?</div>
<div class="lede">We forecast the NY unemployment rate 1, 3 and 6 months ahead using jobless claims, payrolls,
neighboring states and national data, and test every model using only the data that was actually published on
each forecast date.</div>
<div class="kpi-grid">
  <div class="kpi"><div class="label">Latest published rate · {latest['date']:%b %Y}</div>
    <div class="value">{latest['ny_ur']:.1f}%</div><div class="note">Down from {prior['ny_ur']:.1f}% in {prior['date']:%B}</div></div>
  <div class="kpi dark"><div class="label">Our forecast · Sep 2026</div>
    <div class="value">{combo_sep['yhat']:.1f}%</div>
    <div class="note">95% range {combo_sep['lower']:.1f}–{combo_sep['upper']:.1f}% · actual released 20 Oct</div></div>
  <div class="kpi"><div class="label">How we tested</div>
    <div class="value">441</div>
    <div class="note">monthly forecast dates, 1990–2026, each using only data published at the time</div></div>
  <div class="kpi"><div class="label">Why Combo · error vs. “no change”</div>
    <div class="cmp">
      <span></span><span class="h">Best</span><span class="h">Average</span><span class="h">Worst</span>
      <span class="m">Combo</span><span class="v good">−{abs(combo['min']):.0%}</span><span class="v good">−{abs(combo['mean']):.0%}</span><span class="v bad">+{combo['max']:.0%}</span>
      <span class="m">ML</span><span class="v good">−{abs(ml['min']):.0%}</span><span class="v good">−{abs(ml['mean']):.0%}</span><span class="v bad">+{ml['max']:.0%}</span>
    </div>
    <div class="note">ML wins more on average; Combo never loses badly. 3 test periods × 3 horizons.</div></div>
</div>
""", unsafe_allow_html=True)

    left, right = st.columns([2.15, 1], gap="medium")

    with left, st.container(border=True):
        st.markdown('<div class="card-title"><div class="t">Forecast explorer</div>'
                    '<span>Backtest forecasts vs. what actually happened</span></div>', unsafe_allow_html=True)
        chosen = st.pills("Models", MODELS, selection_mode="multi", default=["combo_s", "arx", "naive"],
                         format_func=MODEL_LABELS.get) or ["combo_s"]
        b, c, _ = st.columns([1.4, 2, 1])
        horizon = b.segmented_control("Horizon", [1, 3, 6], default=3, format_func=lambda h: f"{h} mo") or 3
        period = c.selectbox("Period", list(PERIODS), format_func=PERIODS.get)

        st.plotly_chart(charts.explorer_chart(forecasts, panel, chosen, horizon, PERIOD_WINDOWS[period]),
                        width="stretch", config={"displayModeBar": False})

        scores = metrics[(metrics["view"] == period) & (metrics["horizon"] == horizon)].set_index("model")
        stats = "".join(
            f'<div><div class="label">{MODEL_LABELS[m]} vs. naive</div>'
            f'<div class="value">{scores.loc[m, "rmse_vs_naive"]:.2f}×</div></div>'
            for m in chosen if m in scores.index and m != "naive")
        stats += (f'<div><div class="label">Naive RMSE</div><div class="value">{scores.loc["naive", "rmse"]:.3f} pp</div></div>'
                  f'<div><div class="label">Months scored</div><div class="value">{int(scores.loc["naive", "n"])}</div></div>')
        st.markdown(f'<div class="stats">{stats}</div>', unsafe_allow_html=True)
        st.caption("Each point is the forecast made *h* months earlier, using only data published at that time. "
                   "Dots on the right are live forecasts; the band is the Combo 95% range.")

    with right, st.container(border=True):
        st.markdown('<div class="ai-head"><div><div class="t">Ask the forecast</div>'
                    '<div class="sub">Answers from our results, not the open web</div></div>'
                    '<div class="badge">Claude</div></div>', unsafe_allow_html=True)
        enabled = bool(get_api_key())
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat = st.container(height=430, border=False, autoscroll=True, key="chat")
        picked = None
        for text in SUGGESTIONS:
            if st.button(text, key=f"suggest_{text}", disabled=not enabled):
                picked = text
        typed = st.chat_input("Ask about trends, models or forecasts…" if enabled
                              else "Preview: add an API key to enable", disabled=not enabled)

        with chat:
            if not enabled:
                for role, text in PREVIEW_CHAT:
                    st.markdown(f'<div class="bubble {role}">{text}</div>', unsafe_allow_html=True)
                st.markdown('<div class="preview-note">Preview conversation. Set ANTHROPIC_API_KEY to turn the '
                            'assistant on.</div>', unsafe_allow_html=True)
            else:
                if not st.session_state.messages:
                    st.markdown('<div class="bubble ai">Hi! Ask me anything about the NY unemployment forecasts, '
                                'the models, or how we tested them.</div>', unsafe_allow_html=True)
                for message in st.session_state.messages:
                    role = "user" if message["role"] == "user" else "ai"
                    if role == "user":
                        st.markdown(f'<div class="bubble user">{html.escape(message["content"])}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])
                question = typed or picked
                if question:
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.markdown(f'<div class="bubble user">{html.escape(question)}</div>', unsafe_allow_html=True)
                    answer = st.write_stream(stream_answer(st.session_state.messages))
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.container(border=True):
        st.markdown('<div class="card-title"><div class="t">Live forecasts</div>'
                    '<span>Made 30 Sep 2026 with August data (September target: made 31 Aug)</span></div>',
                    unsafe_allow_html=True)
        table = live.pivot_table(index=["date", "horizon"], columns="model", values="yhat").reset_index()
        combo = live[live["model"] == "combo_s"].set_index(["date", "horizon"])
        table["Combo 95% range"] = [f"{combo.loc[(d, h), 'lower']:.2f}–{combo.loc[(d, h), 'upper']:.2f}"
                                    for d, h in zip(table["date"], table["horizon"])]
        table["Target"] = table["date"].dt.strftime("%b %Y") + " · " + table["horizon"].astype(str) + " mo"
        order = ["Target", "combo_s", "Combo 95% range"] + [m for m in MODELS if m != "combo_s"]
        st.dataframe(table[order].rename(columns=MODEL_LABELS), hide_index=True, width="stretch",
                     column_config={MODEL_LABELS[m]: st.column_config.NumberColumn(format="%.2f") for m in MODELS})
        st.caption("Ranges come from errors in the calm 2016–19 validation years, so they understate the risk of a shock.")

    findings = [
        ("Indicators help at turning points",
         "During the 2022 decline, claims, payrolls and neighboring states cut 3-month error by about a third "
         "versus “no change”."),
        ("In calm times, “no change” is hard to beat",
         "From 2023 on, models read noise as trend. At 6 months nothing reliably beats the latest value."),
        ("Nobody forecast COVID",
         "The rate jumped to 16.7% in May 2020. Every model’s error was about as large as naive’s: shocks need "
         "different tools."),
    ]
    for col, (i, (title, body)) in zip(st.columns(3), enumerate(findings, start=1)):
        col.markdown(f'<div class="finding"><div class="num">0{i}</div><h4>{title}</h4><p>{body}</p></div>',
                     unsafe_allow_html=True)

# ---------------------------------------------------------------- Model scorecard
with scorecard:
    st.markdown('<div class="eyebrow">Model scorecard</div>'
                '<div class="hero">Which model wins depends on whether the job market is turning</div>'
                '<div class="lede">Each bar is a model’s forecast error divided by the error of “no change”. '
                'Blue beats naive; red loses to it.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        a, b = st.columns([2, 1])
        view = a.segmented_control("Period", list(PERIODS), default="post_covid", format_func=PERIODS.get,
                                   key="score_period") or "post_covid"
        h = b.segmented_control("Horizon", [1, 3, 6], default=3, format_func=lambda x: f"{x} mo",
                                key="score_h") or 3
        takeaways = {
            "post_covid": "While unemployment was falling after COVID, indicator models (ARX, VAR, ML) cut error by up to a third.",
            "calm_2023on": "Once things settled, the same models over-reacted: ARX is 1.21–1.66× naive at 3–6 months. Shrunk models stay safe.",
            "covid": "No model anticipated the shutdown. Errors were about as large as naive's, and VAR was worse.",
            "validation": "Settings were chosen on these years, so these scores are optimistic for the shrunk models.",
        }
        st.info(takeaways[view])
        st.plotly_chart(charts.scorecard_chart(metrics, view, h), width="stretch", config={"displayModeBar": False})
        st.caption("★ The only statistically significant win vs. naive: Combo at 1 month, 2022–26 (Diebold-Mariano "
                   "p = 0.02). With about 50 test months, most other gaps could be noise.")

    with st.container(border=True):
        st.markdown('<div class="card-title"><div class="t">Do indicators add information?</div>'
                    '<span>Same Ridge model, with and without the indicators</span></div>', unsafe_allow_html=True)
        st.write("Bars show how much lower the error is than a Ridge that only knows the unemployment rate's own history.")
        ab_period = st.segmented_control("Period", ["validation", "post_covid"], default="post_covid",
                                         format_func=PERIODS.get, key="ab_period") or "post_covid"
        st.plotly_chart(charts.ablation_chart(load_ablation(), ab_period), width="stretch",
                        config={"displayModeBar": False})

    with st.expander("Full accuracy table"):
        full = metrics[["view", "horizon", "model", "n", "rmse", "mae", "rmse_vs_naive", "dm_p_vs_naive"]].copy()
        full["view"] = full["view"].map(PERIODS)
        full["model"] = full["model"].map(MODEL_LABELS)
        st.dataframe(full, hide_index=True, width="stretch")

# ---------------------------------------------------------------- Indicators (old overlay view, hidden)
if SHOW_OLD_INDICATORS:
    with tabs[4]:
        st.markdown('<div class="eyebrow">Indicators</div>'
                    '<div class="hero">50 years of NY unemployment, and what moves with it</div>', unsafe_allow_html=True)
        dictionary = load_dictionary()
        columns = dictionary[(dictionary["file"] == "monthly_panel.csv") & (dictionary["name"] != "ny_ur")
                             & (dictionary["name"] != "date")]
        options = ["None"] + columns["name"].tolist()
        describe = dict(zip(columns["name"], columns["description"]))
        with st.container(border=True):
            overlay = st.selectbox("Overlay an indicator", options,
                                   format_func=lambda n: "None" if n == "None" else f"{n} · {describe[n]}")
            if overlay == "None":
                st.plotly_chart(charts.history_chart(panel), width="stretch", config={"displayModeBar": False})
            else:
                st.plotly_chart(charts.history_chart(panel, overlay, overlay), width="stretch",
                                config={"displayModeBar": False})
                row = columns[columns["name"] == overlay].iloc[0]
                st.caption(f"{row['description']} · {row['unit']} · {row['source']} · from {row['start_date']} · "
                           f"published {row['typical_release_lag']}")

        st.markdown("#### Alternative data")
        st.write("Indeed job postings (from 2020) and NY taxable sales (quarterly, from 2013) are in the data panel but "
                 "not in the final models: their history is too short or too infrequent to test fairly on the 2016–19 "
                 "validation years. They are shown here as context only.")
        with st.expander("Data dictionary"):
            st.dataframe(dictionary, hide_index=True, width="stretch")

# ---------------------------------------------------------------- Methodology
with method:
    st.markdown('<div class="eyebrow">Methodology</div>'
                '<div class="hero">How we tested: no peeking at the future</div>', unsafe_allow_html=True)
    st.markdown("""
**Pseudo-real-time backtest.** Every forecast is made on the last day of a month using only numbers that had been
published by that day, based on a calendar of real release dates. On 31 August 2026, July's rate (4.4%) was known
but August's was not, so a "1-month" forecast of September is really two steps past the last known value.

| Period | Years | Used for |
|---|---|---|
| Train | 1990–2015 | Fitting the models (refit every month on all data so far) |
| Validation | 2016–2019 | Choosing settings, which are then locked |
| COVID test | 2020–2021 | Shock test |
| Post-COVID test | 2022–Aug 2026 | Main test |

**Two fixes that changed the results.**
1. *Release timing.* The framework first assumed state data came out 45 days after month end. Matching 72 verified
   BLS release dates (about 3 weeks) gave every model last month's rate, which it really had. Naive's 1-month error
   fell about 28%.
2. *COVID in training.* A tenfold claims spike dominated the regressions (one forecast was 71% unemployment). Dropping
   2020–21 from training (it stays in scoring) took ARX from 2.7–3.7× naive's error to 0.65–0.80×.

**Shrinkage.** Forecast = naive + λ × (model − naive). Validation chose λ near 1 at 1 month and near 0 at 6 months:
the further out, the less the models are trusted.
""")
    st.markdown("#### Models")
    st.dataframe(pd.DataFrame({"Model": [MODEL_LABELS[m] for m in MODELS],
                               "What it uses": [MODEL_DESCRIPTIONS[m] for m in MODELS]}),
                 hide_index=True, width="stretch")
    st.markdown("""
#### Limitations
- “Pseudo” real time: the team had already seen 2020–26, and data values are today's revised versions.
- Uncertainty ranges are calibrated on calm years and understate shock risk.
- About 50 test months: most differences between models are not statistically significant.
- October 2025 has no unemployment rate (federal shutdown) and is never scored.

#### How we used AI
[TEAM TO FILL IN: e.g. Claude Code for pipeline, modeling and dashboard code; the Claude API powers the assistant.]
""")

# ---------------------------------------------------------------- Indicators, alternative tile view
# (name, panel column, how it is shown, role in the models, short note)
TILE_GROUPS = [
    ("Used in the final models", "used", "Used", [
        ("NY jobless claims", "ny_claims_avg", "{:,.0f}", "Monthly average of weekly initial claims, NSA"),
        ("NY payroll jobs", "ny_payrolls", "{:,.0f}k", "Nonfarm employees, thousands, SA"),
        ("US unemployment rate", "us_ur", "{:.1f}%", "National rate, SA"),
        ("NJ unemployment rate", "nj_ur", "{:.1f}%", "Used in VAR and regional ML"),
        ("CT unemployment rate", "ct_ur", "{:.1f}%", "Used in VAR and regional ML"),
        ("PA unemployment rate", "pa_ur", "{:.1f}%", "Used in VAR and regional ML"),
        ("Empire State survey", "empire_general", "{:+.1f}", "NY Fed manufacturing conditions; regional ML only"),
    ]),
    ("Tried in the first pass only", "first", "First pass only", [
        ("NY coincident index", "ny_coincident", "{:.1f}", "Philly Fed index of NY activity"),
        ("US jobless claims", "us_claims", "{:,.0f}", "Week containing the 12th, SA"),
        ("Fed funds rate", "fedfunds", "{:.2f}%", "Effective rate"),
        ("10y–3m Treasury spread", "spread_10y3m", "{:+.2f} pp", "Yield curve, monthly average"),
        ("S&P 500", "sp500", "{:,.0f}", "Monthly average close"),
    ]),
]

def placeholder_trends():
    """Simulated Google Trends-style series (0-100) for the layout only. NOT real data: replace when collected."""
    dates = pd.date_range("2016-01-01", "2026-09-01", freq="MS")
    months = np.arange(len(dates))
    interest = 22 + 4 * np.sin(2 * np.pi * months / 12) + np.random.default_rng(0).normal(0, 1.5, len(dates))
    covid = dates.get_loc(pd.Timestamp("2020-04-01"))
    interest[covid:] += 75 * np.exp(-(months[covid:] - covid) / 3)
    return pd.DataFrame({"date": dates, "interest": interest.clip(0, 100)})


with indicators_alt:
    st.markdown('<div class="eyebrow">Indicators</div>'
                '<div class="hero">Every series in our panel, and what the models did with it</div>'
                '<div class="lede">Each tile shows the last ten years, the latest value and its change over a year. '
                'The tag says whether the final models use it.</div>', unsafe_allow_html=True)

    def tile(col, name, column, fmt, note, tag_class, tag_text, since="2016-01-01"):
        series = panel[["date", column]].dropna()
        last, year_ago = series.iloc[-1], series.iloc[-13] if len(series) > 13 else series.iloc[0]
        # Rates and indexes change in points; levels (claims, jobs, prices) in percent.
        if "%" in fmt or "pp" in fmt or column == "empire_general":
            change = f"{last[column] - year_ago[column]:+.2f} pts"
        else:
            change = f"{last[column] / year_ago[column] - 1:+.1%}"
        with col, st.container(border=True):
            st.markdown(f'<div class="tile-name">{name}</div><div class="tile-meta">{note}</div>'
                        f'<span class="tag {tag_class}">{tag_text}</span>', unsafe_allow_html=True)
            recent = series[series["date"] >= since]
            st.plotly_chart(charts.sparkline(recent["date"], recent[column]), width="stretch",
                            config={"displayModeBar": False}, key=f"spark_{column}")
            st.markdown(f'<div class="tile-value">{fmt.format(last[column])}</div>'
                        f'<div class="tile-change">{last["date"]:%b %Y} · {change} vs. a year earlier</div>',
                        unsafe_allow_html=True)

    for title, tag_class, tag_text, tiles in TILE_GROUPS:
        st.markdown(f'<div class="group-title">{title}</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (name, column, fmt, note) in enumerate(tiles):
            tile(cols[i % 4], name, column, fmt, note, tag_class, tag_text)

    st.markdown('<div class="group-title">Alternative data · context only</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    tile(a, "Indeed job postings", "indeed_postings", "{:.1f}", "NY postings index, Feb 2020 = 100",
         "context", "Too short to test", since="2020-02-01")
    taxable = load_taxable_sales()
    with b, st.container(border=True):
        last = taxable.iloc[-1]
        st.markdown('<div class="tile-name">NY taxable sales</div><div class="tile-meta">Quarterly, seasonally '
                    'adjusted, from 2013</div><span class="tag context">Quarterly, not modeled</span>',
                    unsafe_allow_html=True)
        st.plotly_chart(charts.sparkline(taxable["date"], taxable["taxable_sales_sa"] / 1e9), width="stretch",
                        config={"displayModeBar": False}, key="spark_taxable")
        st.markdown(f'<div class="tile-value">${last["taxable_sales_sa"] / 1e9:,.0f}B</div>'
                    f'<div class="tile-change">{last["selling_period"]} {last["date"]:%Y}</div>',
                    unsafe_allow_html=True)
    with c, st.container(border=True):
        st.markdown('<div class="tile-name">Google search trends</div><div class="tile-meta">Searches for '
                    '“unemployment”, “file for unemployment” in NY</div><span class="tag idea">Placeholder · '
                    'simulated, not real data</span>', unsafe_allow_html=True)
        trends = placeholder_trends()
        st.plotly_chart(charts.sparkline(trends["date"], trends["interest"], color=charts.MUTED), width="stretch",
                        config={"displayModeBar": False}, key="spark_trends")
        st.markdown('<div class="tile-change" style="font-size:12px;line-height:1.45">Real series available '
                    'monthly from 2004, so it could be tested on the 2016–19 validation years. Swap in Google '
                    'Trends data to replace this placeholder.</div>', unsafe_allow_html=True)

    st.markdown('<div class="group-title">Compare with NY unemployment</div>', unsafe_allow_html=True)
    with st.container(border=True):
        choices = {column: name for _, _, _, tiles in TILE_GROUPS for name, column, _, _ in tiles}
        choices = {"indeed_postings": "Indeed job postings", **choices}
        a, b = st.columns([2, 1.4])
        compare = a.selectbox("Indicator", list(choices), format_func=choices.get, key="compare_col")
        window = b.segmented_control("Window", ["2020", "2010", "1990", "1976"], default="2020",
                                     format_func=lambda y: "All" if y == "1976" else f"Since {y}",
                                     key="compare_window") or "2020"
        st.plotly_chart(charts.paired_chart(panel, compare, choices[compare], f"{window}-01-01"), width="stretch",
                        config={"displayModeBar": False})
        if compare == "indeed_postings":
            st.caption("Postings collapsed in spring 2020 as unemployment spiked, peaked in 2022 as it fell, and "
                       "have drifted down since. Only about six years of history, mostly one cycle, so it was not "
                       "added to the models.")
        else:
            st.caption("Each series has its own axis; read the timing of turns, not the levels.")
