import hashlib
import io
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from shopcast import llm  # noqa: E402
from shopcast.agent import ShopContext, ask  # noqa: E402
from shopcast.anomaly import anomaly_table, find_anomalies  # noqa: E402
from shopcast.features import load_sales  # noqa: E402
from shopcast.forecast import Forecast, backend, backtest, forecast  # noqa: E402

SAMPLE = Path(__file__).parent / "sample_data" / "lagos_bakery.csv"
CURRENCY = os.getenv("CURRENCY_SYMBOL", "₦")
MODEL = "TabPFN" if backend() == "tabpfn" else "the stand-in model"

st.set_page_config(page_title="ShopCast", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False, max_entries=8)
def analyse(csv_bytes: bytes, horizon: int, model_backend: str) -> dict:
    history = load_sales(io.BytesIO(csv_bytes))
    return {
        "history": history,
        "forecast": forecast(history, horizon).frame,
        "scored": find_anomalies(history),
        "backtest": backtest(history, 28),
    }


def money(v: float) -> str:
    return f"{CURRENCY}{v:,.0f}"


def forecast_chart(history: pd.DataFrame, fc: pd.DataFrame) -> alt.Chart:
    past = history.tail(90).assign(kind="Actual").rename(columns={"sales": "value"})
    future = fc.assign(kind="Forecast").rename(columns={"forecast": "value"})
    lines = pd.concat([past[["date", "value", "kind"]], future[["date", "value", "kind"]]])
    chart = (
        alt.Chart(lines)
        .mark_line()
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title=f"Daily sales ({CURRENCY})"),
            color=alt.Color(
                "kind:N", title=None, scale=alt.Scale(domain=["Actual", "Forecast"], range=["#4c9be8", "#f5a623"])
            ),
            strokeDash=alt.condition(alt.datum.kind == "Forecast", alt.value([5, 3]), alt.value([0])),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=",.0f"), "kind:N"],
        )
    )
    if fc["low"].notna().any():
        band = alt.Chart(fc).mark_area(opacity=0.2, color="#f5a623").encode(x="date:T", y="low:Q", y2="high:Q")
        chart = band + chart
    return chart.properties(height=320)


def backtest_chart(frame: pd.DataFrame) -> alt.Chart:
    long = frame.melt("date", var_name="series", value_name="sales")
    long["series"] = long["series"].map({"actual": "Actual", "model": MODEL, "naive": "Same day last week"})
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("sales:Q", title=f"Sales ({CURRENCY})"),
            color=alt.Color("series:N", title=None),
            tooltip=["series:N", alt.Tooltip("date:T"), alt.Tooltip("sales:Q", format=",.0f")],
        )
        .properties(height=280)
    )


with st.sidebar:
    st.title("📈 ShopCast")
    st.caption("Sales forecasts for small shops, powered by open models: TabPFN for the numbers, Gemma for the words.")
    shop_name = st.text_input("Shop name", placeholder="e.g. Mama Tolu's Bakery") or "Your shop"
    upload = st.file_uploader("Daily sales CSV", type="csv", help="Columns: date, sales, and optionally promo (0/1).")
    use_sample = st.toggle("Use sample bakery data instead", value=False)
    horizon = st.slider("Forecast horizon (days)", 7, 56, 28, step=7)

    if use_sample:
        csv_bytes = SAMPLE.read_bytes()
    elif upload is not None:
        csv_bytes = upload.getvalue()
    else:
        csv_bytes = None

    run_key = (hashlib.sha256(csv_bytes).hexdigest(), horizon) if csv_bytes else None
    if st.button("Start forecast", type="primary", disabled=csv_bytes is None, width="stretch"):
        st.session_state["run_key"] = run_key

    st.divider()
    st.caption(f"Forecast model: **{'TabPFN (Prior Labs API)' if backend() == 'tabpfn' else 'offline sklearn stand-in'}**")
    st.caption(f"Language model: **{llm.model_name()}** via {llm.provider()}")

if csv_bytes is None:
    st.header("Welcome to ShopCast")
    st.info("Upload a daily sales CSV with `date` and `sales` columns (and optionally `promo`), or switch on the sample data, then click **Start forecast**.")
    st.stop()

if st.session_state.get("run_key") != run_key:
    st.header(shop_name)
    try:
        preview = load_sales(io.BytesIO(csv_bytes))
    except Exception as exc:
        st.error(f"This file can't be used yet: {exc}")
        st.stop()
    st.success(
        f"File looks good: {len(preview)} days of sales from {preview['date'].min():%d %b %Y} "
        f"to {preview['date'].max():%d %b %Y}"
        + (", with promo days." if preview["promo"].any() else ". No promo column, so the what-if tool is off.")
    )
    st.dataframe(preview.tail(10).assign(date=preview["date"].dt.strftime("%a %d %b %Y")), hide_index=True, width="stretch")
    st.info(f"Click **Start forecast** in the sidebar. The first run on a file takes a few minutes while {MODEL} studies the history.")
    st.stop()

try:
    with st.spinner(f"Asking {MODEL} about your sales history. This can take a few minutes on the first run..."):
        data = analyse(csv_bytes, horizon, backend())
except Exception as exc:  # surface config and data problems to the owner
    st.error(f"Could not analyse this file: {exc}")
    st.stop()

history, fc_frame, scored, bt = data["history"], data["forecast"], data["scored"], data["backtest"]

st.header(shop_name)
last = history.tail(horizon)["sales"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Forecast, next {horizon} days", money(fc_frame["forecast"].sum()), f"{(fc_frame['forecast'].sum() / last - 1) * 100:+.1f}% vs last {horizon}")
c2.metric("Average day (history)", money(history["sales"].mean()))
c3.metric("Unusual days found", int(scored["is_anomaly"].sum()))
c4.metric("Backtest error vs naive", f"{bt['improvement_pct']:.0f}% lower" if bt["improvement_pct"] > 0 else f"{-bt['improvement_pct']:.0f}% higher")

tab_fc, tab_anom, tab_bt, tab_chat = st.tabs(["Forecast", "Unusual days", "How accurate is it?", "Ask ShopCast"])

with tab_fc:
    st.altair_chart(forecast_chart(history, fc_frame), width="stretch")
    weekly = fc_frame.groupby(fc_frame.index // 7).agg(week_starting=("date", "min"), forecast=("forecast", "sum"))
    weekly["week_starting"] = weekly["week_starting"].dt.strftime("%a %d %b %Y")
    weekly["forecast"] = weekly["forecast"].map(money)
    weekly.columns = ["Week starting", "Forecast sales"]
    st.dataframe(weekly, hide_index=True, width="stretch")

with tab_anom:
    table = anomaly_table(scored, top_n=15)
    if table.empty:
        st.success("No unusual days: sales stayed close to what the model expected.")
    else:
        st.write(f"Days where sales were far from what {MODEL} expected, given the weekday, payday, promos and recent trend.")
        st.dataframe(
            table.assign(date=table["date"].dt.strftime("%a %d %b %Y"), sales=table["sales"].map(money), expected=table["expected"].map(money)),
            hide_index=True,
            width="stretch",
        )

with tab_bt:
    st.write(
        f"ShopCast hid the last {bt['days']} days, forecast them, and compared with the simple rule "
        f"\"same as the same weekday last week\"."
    )
    m1, m2 = st.columns(2)
    m1.metric(f"{MODEL[0].upper()}{MODEL[1:]} average daily error", money(bt["model_mae"]), f"{bt['model_mape']:.1f}% MAPE", delta_color="off")
    m2.metric("Naive average daily error", money(bt["naive_mae"]), f"{bt['naive_mape']:.1f}% MAPE", delta_color="off")
    st.altair_chart(backtest_chart(bt["frame"]), width="stretch")

with tab_chat:
    st.caption("Try: \"How much will I sell in the next two weeks?\", \"Why was there a drop in September?\", \"What if I run a promo every Friday?\"")
    key = f"chat-{hash(csv_bytes)}"
    st.session_state.setdefault(key, [])

    def show_facts(tool: str | None, facts: dict | None) -> None:
        if tool:
            st.caption(tool)
        if facts:
            with st.expander("Numbers behind this answer"):
                st.json(facts)

    thread = st.container()
    with thread:
        for msg in st.session_state[key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                show_facts(msg.get("tool"), msg.get("facts"))

    if question := st.chat_input("Ask about your sales"):
        st.session_state[key].append({"role": "user", "content": question})
        with thread, st.chat_message("user"):
            st.markdown(question)
        with thread, st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ctx = ShopContext(
                    history=history,
                    base_forecast=Forecast(fc_frame),
                    scored=scored,
                    backtest=bt,
                    currency=CURRENCY,
                    shop_name=shop_name,
                )
                try:
                    reply = ask(ctx, question)
                    tool = f"Tool used: {reply.action} {reply.args or ''}"
                    content, facts = reply.text, reply.result
                except Exception as exc:
                    tool, facts = None, None
                    content = f"Sorry, I could not reach the language model: {exc}"
            st.markdown(content)
            show_facts(tool, facts)
        st.session_state[key].append({"role": "assistant", "content": content, "tool": tool, "facts": facts})
