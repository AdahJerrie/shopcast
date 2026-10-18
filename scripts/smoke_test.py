"""Run the full pipeline on the sample data with a stubbed LLM.

Uses whatever FORECAST_BACKEND is set; `FORECAST_BACKEND=sklearn` runs fully offline.
Pass --live-llm to call the configured Gemma provider instead of the stub.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from shopcast import llm
from shopcast.agent import ShopContext, ask
from shopcast.anomaly import anomaly_table, find_anomalies
from shopcast.features import load_sales
from shopcast.forecast import backend, backtest, forecast

start = time.time()
history = load_sales(Path(__file__).resolve().parents[1] / "sample_data" / "lagos_bakery.csv")
print(f"backend={backend()} rows={len(history)}")

fc = forecast(history, 28)
print(fc.frame.head(7).round(0).to_string())
print(f"28-day total: {fc.total:,.0f}")

bt = backtest(history, 28)
print({k: round(v, 1) for k, v in bt.items() if k != "frame"})

scored = find_anomalies(history)
print(anomaly_table(scored, 6).to_string())

ctx = ShopContext(history, fc, scored, bt, shop_name="Sample bakery")
questions = [
    "What if I run a promo every Friday?",
    "Which days looked strange?",
    "How much will I sell in the next two weeks?",
    "How accurate is this?",
]


def stub(prompt: str) -> str:
    if "Reply with only a JSON object" in prompt:
        return 'Sure! {"action": "explain", "args": {}}'
    return "stub answer"


generate = llm.generate if "--live-llm" in sys.argv else stub

for q in questions:
    reply = ask(ctx, q, generate=generate)
    print(f"\nQ: {q}\naction={reply.action} args={reply.args}")
    print("result:", json.dumps(reply.result, default=str)[:400])
    print("answer:", reply.text)

print(f"\nDone in {time.time() - start:.1f}s")
