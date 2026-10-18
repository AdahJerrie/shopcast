"""A small prediction agent: Gemma picks a tool, ShopCast runs it, Gemma explains the result."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from shopcast import llm
from shopcast.anomaly import anomaly_table
from shopcast.features import has_promo
from shopcast.forecast import Forecast, forecast

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MAX_DAYS = 56


@dataclass
class ShopContext:
    history: pd.DataFrame
    base_forecast: Forecast
    scored: pd.DataFrame
    backtest: dict
    currency: str = "₦"
    shop_name: str = "the shop"
    extra_forecasts: dict = field(default_factory=dict)


@dataclass
class AgentReply:
    text: str
    action: str
    args: dict
    result: dict


def _money(ctx: ShopContext, value: float) -> str:
    return f"{ctx.currency}{value:,.0f}"


def overview(ctx: ShopContext) -> dict:
    h = ctx.history
    last28 = h.tail(28)
    prev28 = h.iloc[-56:-28]
    by_weekday = h.groupby(h["date"].dt.dayofweek)["sales"].mean()
    return {
        "shop": ctx.shop_name,
        "history": f"{h['date'].min():%d %b %Y} to {h['date'].max():%d %b %Y} ({len(h)} days)",
        "avg_daily_sales": _money(ctx, h["sales"].mean()),
        "last_28_days_total": _money(ctx, last28["sales"].sum()),
        "previous_28_days_total": _money(ctx, prev28["sales"].sum()),
        "best_weekday": WEEKDAYS[int(by_weekday.idxmax())],
        "slowest_weekday": WEEKDAYS[int(by_weekday.idxmin())],
        "has_promo_data": has_promo(h),
        "next_28_days_forecast_total": _money(ctx, ctx.base_forecast.total),
        "anomalies_found": int(ctx.scored["is_anomaly"].sum()),
        "backtest": {
            "model_mae": _money(ctx, ctx.backtest["model_mae"]),
            "naive_mae": _money(ctx, ctx.backtest["naive_mae"]),
            "improvement_vs_naive_pct": round(ctx.backtest["improvement_pct"], 1),
        },
    }


def _forecast_frame(ctx: ShopContext, days: int) -> pd.DataFrame:
    if days <= len(ctx.base_forecast.frame):
        return ctx.base_forecast.frame.head(days)
    if days not in ctx.extra_forecasts:
        ctx.extra_forecasts[days] = forecast(ctx.history, days)
    return ctx.extra_forecasts[days].frame


def tool_forecast(ctx: ShopContext, days: int = 28) -> dict:
    days = max(1, min(int(days), MAX_DAYS))
    frame = _forecast_frame(ctx, days)
    weekly = frame.groupby(frame.index // 7).agg(start=("date", "min"), total=("forecast", "sum"))
    return {
        "days": days,
        "total": _money(ctx, frame["forecast"].sum()),
        "daily_average": _money(ctx, frame["forecast"].mean()),
        "busiest_day": f"{frame.loc[frame['forecast'].idxmax(), 'date']:%a %d %b}",
        "quietest_day": f"{frame.loc[frame['forecast'].idxmin(), 'date']:%a %d %b}",
        "weekly_totals": [
            {"week_starting": f"{r.start:%d %b}", "total": _money(ctx, r.total)} for r in weekly.itertuples()
        ],
    }


def tool_anomalies(ctx: ShopContext, top_n: int = 5) -> dict:
    table = anomaly_table(ctx.scored, top_n=max(1, min(int(top_n), 15)))
    return {
        "count": int(ctx.scored["is_anomaly"].sum()),
        "days": [
            {
                "date": f"{r.date:%a %d %b %Y}",
                "sales": _money(ctx, r.sales),
                "expected": _money(ctx, r.expected),
                "difference_pct": r.pct_vs_expected,
                "type": r.direction,
                "promo_day": bool(r.promo),
            }
            for r in table.itertuples()
        ],
    }


def tool_what_if(
    ctx: ShopContext,
    promo_weekdays: list | None = None,
    promo_dates: list | None = None,
    days: int = 28,
) -> dict:
    if not has_promo(ctx.history):
        return {"error": "This sales file has no promo column, so ShopCast cannot measure promo effects."}

    days = max(7, min(int(days), MAX_DAYS))
    start = ctx.history["date"].max() + pd.Timedelta(days=1)
    window = pd.date_range(start, periods=days, freq="D")
    weekdays = {int(w) for w in (promo_weekdays or []) if str(w).lstrip("-").isdigit() and 0 <= int(w) <= 6}
    dates = {pd.Timestamp(d).normalize() for d in (promo_dates or []) if _is_date(d)}
    chosen = {d for d in window if d.dayofweek in weekdays or d in dates}
    if not chosen:
        return {"error": "No promo days fell inside the forecast window. Name weekdays or dates in the next few weeks."}

    base_total = float(_forecast_frame(ctx, days)["forecast"].sum())
    promo_total = forecast(ctx.history, days, promo_dates=chosen).total
    return {
        "days": days,
        "promo_days": [f"{d:%a %d %b}" for d in sorted(chosen)],
        "forecast_without_promo": _money(ctx, base_total),
        "forecast_with_promo": _money(ctx, promo_total),
        "difference": _money(ctx, promo_total - base_total),
        "difference_pct": round((promo_total / base_total - 1) * 100, 1) if base_total else None,
        "note": "Revenue only; subtract the cost of the discount to judge profit.",
    }


def _is_date(value) -> bool:
    try:
        pd.Timestamp(value)
        return True
    except (TypeError, ValueError):
        return False


TOOLS: dict[str, Callable[..., dict]] = {
    "forecast": tool_forecast,
    "anomalies": tool_anomalies,
    "what_if": tool_what_if,
}

ROUTER_PROMPT = """You are ShopCast, a sales assistant for a small shop. Pick ONE action for the owner's question.

Actions:
- "forecast": predict future sales. args: {{"days": 1-56}}
- "anomalies": list unusual sales days in the past. args: {{"top_n": 1-15}}
- "what_if": estimate the effect of running promos. args: {{"promo_weekdays": [0-6, Monday=0], "promo_dates": ["YYYY-MM-DD"], "days": 7-56}}
- "explain": answer from the shop overview alone (trends, best days, how accurate the model is). args: {{}}

Today is the day after the last date in the history. Shop overview:
{overview}

Owner's question: {question}

Reply with only a JSON object like {{"action": "forecast", "args": {{"days": 14}}}}. No other text."""

ANSWER_PROMPT = """You are ShopCast, a friendly sales assistant for a small shop owner who is not a data person.
Answer their question using ONLY the facts below. Copy amounts exactly as written, including the {currency} sign.
Never invent percentages or amounts that are not listed. Keep it under 120 words and end with one practical suggestion.

Shop overview:
{overview}

Result of the "{action}" tool:
{result}

Owner's question: {question}"""

INTENTS = [
    ("what_if", re.compile(r"\bwhat if\b|promo|discount|\boffer\b|\bdeal\b|campaign|price cut", re.I)),
    ("anomalies", re.compile(r"strange|unusual|odd|weird|anomal|spike|drop|\bdip\b|outlier|why (was|were|did)", re.I)),
    ("forecast", re.compile(r"\bnext\b|forecast|predict|will i|expect|coming|tomorrow|this (week|month)", re.I)),
]
NUMBER_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8}
UNIT_DAYS = {"day": 1, "week": 7, "month": 28}


def keyword_intent(question: str) -> str | None:
    for action, pattern in INTENTS:
        if pattern.search(question):
            return action
    return None


def extract_args(question: str) -> dict:
    """Pull horizon, weekdays and ISO dates straight from the question text."""
    q = question.lower()
    args: dict = {}
    m = re.search(r"\b(\d+|a|one|two|three|four|five|six|eight)\s*(day|week|month)s?\b", q)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        args["days"] = n * UNIT_DAYS[m.group(2)]
    elif "tomorrow" in q:
        args["days"] = 1
    elif re.search(r"next week|this week", q):
        args["days"] = 7
    elif re.search(r"next month|this month", q):
        args["days"] = 28

    weekdays = [i for i, name in enumerate(WEEKDAYS) if re.search(rf"\b{name.lower()}s?\b", q)]
    if "weekend" in q:
        weekdays += [5, 6]
    if weekdays:
        args["promo_weekdays"] = sorted(set(weekdays))
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q)
    if dates:
        args["promo_dates"] = dates
    return args


def _call_tool(ctx: ShopContext, action: str, args: dict) -> dict:
    accepted = {
        "forecast": {"days"},
        "anomalies": {"top_n"},
        "what_if": {"promo_weekdays", "promo_dates", "days"},
    }[action]
    clean = {k: v for k, v in args.items() if k in accepted}
    try:
        return TOOLS[action](ctx, **clean)
    except (TypeError, ValueError):
        return TOOLS[action](ctx)


def parse_action(text: str) -> tuple[str, dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            action = str(data.get("action", "explain"))
            args = data.get("args") or {}
            if action in TOOLS or action == "explain":
                return action, args if isinstance(args, dict) else {}
        except json.JSONDecodeError:
            pass
    return "explain", {}


def ask(ctx: ShopContext, question: str, generate: Callable[[str], str] = llm.generate) -> AgentReply:
    """Clear questions are routed by keywords (reliable even on a 1B model); Gemma routes the rest."""
    info = overview(ctx)
    info_json = json.dumps(info, indent=1, default=str, ensure_ascii=False)

    action = keyword_intent(question)
    if action:
        args = extract_args(question)
    else:
        action, args = parse_action(generate(ROUTER_PROMPT.format(overview=info_json, question=question)))
        args = {**args, **extract_args(question)}

    result = {"note": "Answer from the overview."} if action == "explain" else _call_tool(ctx, action, args)

    text = generate(
        ANSWER_PROMPT.format(
            overview=info_json,
            action=action,
            result=json.dumps(result, indent=1, default=str, ensure_ascii=False),
            question=question,
            currency=ctx.currency,
        )
    ).strip()
    return AgentReply(text=text, action=action, args=args, result=result)
