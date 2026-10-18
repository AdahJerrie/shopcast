# ShopCast

Sales forecasts for small shops. Upload a daily sales CSV and ShopCast:

- forecasts the next 1 to 8 weeks with **TabPFN** (Prior Labs' tabular foundation model), with a 10-90% band
- flags unusual days (spikes and drops) by scoring each day with a model that never saw it
- backtests itself on the last 28 days against "same as last week", so the owner can see whether to trust it
- answers questions in plain English through a small **Gemma** agent: forecast, unusual days, and "what if I run a promo every Friday?"

## CSV format

| column | required | notes |
| --- | --- | --- |
| `date` | yes | also accepts `day`, `sale_date`, `transaction_date` |
| `sales` | yes | also accepts `revenue`, `amount`, `total`, `turnover`; multiple rows per day are summed |
| `promo` | no | `1` on promo days; enables the what-if tool |

At least 8 weeks of history is needed. Missing days are treated as zero sales. `sample_data/lagos_bakery.csv` is a year of synthetic bakery sales (regenerate with `python sample_data/generate.py`).

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # then fill in TABPFN_TOKEN and GOOGLE_API_KEY
.\.venv\Scripts\python -m streamlit run app.py
```

- `TABPFN_TOKEN`: sign up at [platform.priorlabs.ai](https://platform.priorlabs.ai) and accept the non-commercial license.
- `GOOGLE_API_KEY`: [Google AI Studio](https://aistudio.google.com/apikey). `GEMMA_MODEL` defaults to `gemma-3-27b-it`.

### Fully offline mode

For a laptop with no internet or no keys:

```powershell
ollama pull gemma3:1b
$env:LLM_PROVIDER="ollama"; $env:FORECAST_BACKEND="sklearn"
.\.venv\Scripts\python -m streamlit run app.py
```

`FORECAST_BACKEND=sklearn` swaps TabPFN for a gradient-boosting stand-in so the app can be developed without an API token; the UI labels it as the stand-in.

`python scripts/smoke_test.py` runs the whole pipeline on the sample data with a stubbed LLM; add `--live-llm` to use the configured Gemma.

## How it works

- `shopcast/features.py`: calendar features (weekday, payday window, Nigerian public holidays, festive season, promo) plus lags of 7 and 14 days and lagged rolling means.
- `shopcast/forecast.py`: because every lag is at least 7 days, a whole week is predicted in one TabPFN call; the forecast walks forward week by week, feeding each week's predictions into the next week's lags.
- `shopcast/anomaly.py`: 5-fold out-of-sample predictions; a day is unusual when its robust z-score exceeds 3 **and** it is at least 25% away from expected.
- `shopcast/agent.py`: clear questions are routed by keywords (reliable even on a 1B model); Gemma routes the rest by returning a JSON action. The tool result is passed back to Gemma with instructions to copy amounts exactly, and the UI shows the raw numbers under every answer.

## Deploy on Render

1. Push this folder to a GitHub repository.
2. In Render, choose **New > Blueprint** and select the repo; `render.yaml` defines a free Python web service.
3. Set `TABPFN_TOKEN` and `GOOGLE_API_KEY` when prompted.
4. Custom domain: claim a free `.tech` domain at [hacktoberfest.com/my/promos](https://hacktoberfest.com/my/promos), add it under the service's **Settings > Custom Domains**, then create the CNAME record Render shows (for example `www` -> `shopcast.onrender.com`) in the .tech DNS panel.

Free Render instances sleep when idle, so the first load after a pause takes about a minute.
