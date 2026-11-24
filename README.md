# Options Analyzer

A read-only options scanner. It pulls option chains from Alpaca, ranks contracts by
visible rules, and puts each stock's earnings history next to what the options are
pricing for the next report. An optional AI brief has Claude write up the setups
worth a closer look.

It only reads market data. It cannot place orders. The ranking is a screen, not a
forecast and not financial advice.

## Try it without any keys

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,ai]"
.venv\Scripts\python.exe -m app serve --demo
```

Open http://127.0.0.1:8000. Demo mode uses synthetic data: the tickers, prices and
earnings dates are made up.

## Use real data

1. Create a free [Alpaca](https://alpaca.markets) account and generate **paper trading** API keys.
2. Copy `.env.example` to `.env` and fill in `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.
3. Set at least one earnings source in `.env`:
   - `SEC_USER_AGENT` (your name and email, which the SEC asks automated clients to send):
     past earnings dates from 8-K filings, free, no daily limit.
   - `ALPHAVANTAGE_API_KEY` (free key, about 25 requests a day): confirmed upcoming dates
     and EPS surprises.
4. Optional: `ANTHROPIC_API_KEY` turns on the AI brief.

```powershell
.venv\Scripts\python.exe -m app serve
```

## Command line

```powershell
.venv\Scripts\python.exe -m app scan                      # discover candidates
.venv\Scripts\python.exe -m app scan AAPL NVDA            # specific symbols
.venv\Scripts\python.exe -m app scan --lens sell --type put --dte-min 20 --dte-max 45
.venv\Scripts\python.exe -m app scan --brief              # add the AI write-up
```

Add `--demo` to any of these to use synthetic data, or `--json` for the full result.

## What a scan does

**Picks symbols.** With symbols given, it scans those. With none, it discovers: your
`WATCHLIST`, the day's most active stocks and biggest movers, and the starter list in
`data/universe.txt`. It keeps up to 20, preferring stocks with earnings within three
weeks and large moves today, and says why each one was picked.

**Measures each contract** in the chosen range of days to expiry:

| Number | Meaning |
|---|---|
| Spread | Bid-ask spread as a share of the mid price |
| Breakeven | Stock move needed at expiry for the buyer to break even |
| vs exp. move | Breakeven distance divided by the one-standard-deviation move the options imply by expiry |
| POP | Probability of profit at expiry, for the buyer or the seller depending on the lens |
| Time decay | Theta as a share of the option's price, per day |
| Annualized yield | Premium over collateral (strike for a put, share price for a call), annualized |

**Adds earnings context** for each stock: the next report date (confirmed from the
calendar, or estimated from the usual gap), the close-to-close move on each of the last
twelve reports, and the size of move the options price for the next one. That last
number takes the first expiration after the report and subtracts ordinary volatility,
which comes from the last expiration before the report, else from the next expiration
after it, else from recent realized volatility.

**Ranks through a lens.** `buy` ranks long calls and puts; `sell` ranks cash-secured
puts and covered calls. The score is a weighted sum of five components (liquidity,
breakeven or probability, how volatility is priced, decay or yield, delta). Every
weight and threshold is in `app/analysis/scoring.py`.

## Limits worth knowing

- **Free Alpaca option quotes are "indicative"**: quotes are modified and trades are
  15 minutes delayed. Screen with them, then check the live quote at your broker.
  Real-time quotes need Alpaca's paid plan (`ALPACA_OPTIONS_FEED=opra`).
- **Free stock quotes come from the IEX exchange only**, a small share of volume.
- **Pricing math is Black-Scholes** with no dividends and a flat rate (`RISK_FREE_RATE`).
  US equity options are American, so deep in-the-money puts and calls just before a
  dividend are approximate.
- **Earnings dates from SEC filings** take their timing from when the 8-K was accepted.
  A company that files hours after its press release can be off by one session.
- **The AI brief knows only the scan.** It has no news and no view of your account. It
  calls `claude-opus-5-5` (override with `AI_MODEL`) and is billed to your Anthropic key.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Roadmap

1. **Done:** read-only scanner with earnings context, discovery and AI brief.
2. News layer: classify headlines and filings by event type, and study what followed
   similar events.
3. Paper trading on Alpaca with a journal that records which signals worked.
4. Live trading with a human approval on every order.
