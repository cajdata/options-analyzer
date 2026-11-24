# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A read-only options scanner. It pulls option chains from Alpaca, ranks contracts by visible rules, and sets each stock's earnings history beside what the options price in. It has no order endpoints and must stay that way: it shows rules and numbers, and the person decides.

## Commands

The project uses the virtual environment in `.venv` (Windows, PowerShell).

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,ai]"                # install
.venv\Scripts\python.exe -m pytest -q                                  # all tests
.venv\Scripts\python.exe -m pytest tests/test_risk.py -q               # one file
.venv\Scripts\python.exe -m pytest tests/test_risk.py::test_the_spread_is_a_cost_to_both_sides -q
.venv\Scripts\python.exe -m app scan --demo                            # command line, synthetic data
.venv\Scripts\python.exe -m app serve --demo                           # dashboard at http://127.0.0.1:8000
```

- `--demo` uses the synthetic provider and needs no keys. Without it the app needs Alpaca keys in `.env` (see `.env.example`).
- No linter is configured. CI (`.github/workflows/tests.yml`) runs pytest on Python 3.11 and 3.13 for every pull request, so code must run on 3.11.
- `.claude/launch.json` defines `options-analyzer-demo`, the demo dashboard on port 8765, for the preview tools. The server does not reload: restart it after changing Python code.

## Architecture

One scan flows in a straight line:

1. **Providers** (`app/providers/`) fetch data. `MarketData` is the protocol; `AlpacaData` is the real one and `DemoMarket` the synthetic one. `EarningsService` combines earnings sources (SEC EDGAR, Alpha Vantage), caches them on disk, and estimates a missing date.
2. **`Scanner`** (`app/scanner.py`) picks symbols (given ones, or discovery from the watchlist, screeners and `data/universe.txt`), then analyzes each on a small thread pool.
3. **Analysis** (`app/analysis/`) runs in a fixed order per symbol: `contracts` (per-contract metrics) → `earnings` (past reactions, implied move) → `risk` (history-based volatility, then risk/reward) → `scoring` (score and rank).
4. **Surfaces** read the resulting `ScanResult`: `app/server.py` (FastAPI plus `public/index.html`), `app/__main__.py` (command line), `app/ai.py` (optional Claude write-up).

Things that are not obvious from any one file:

- **The lens decides late.** `contracts.analyze_contract` produces lens-independent metrics priced at the mid. Only afterwards, in `Scanner._analyze`, does the lens ("buy" or "sell") set `pop`, `outlay`, `score` and `risk_reward`. A `ContractAnalysis` is therefore filled in across several modules.
- **`history_vol` is the shared yardstick.** `risk.history_vol` blends ex-earnings realized volatility with past report-day variance. The score's volatility component and the risk/reward rating both start from it, which is why the rating is shown beside the score and never added into it.
- **Provider caveats travel as notes.** A provider appends to `market.notes`; the scanner drains them into `ScanResult.warnings`, which every surface displays. Use this for anything the person should know (a feed fallback, missing open interest).
- **Time is injected.** `Scanner(today=)`, `EarningsService(today=, clock=)`, `AlpacaData(now=, sleep=)` and `EdgarEarnings(sleep=)` exist so tests never depend on the wall clock. Tests pin `TODAY = date(2026, 11, 24)`.
- **Tunable numbers live in two places.** Score weights and thresholds are in `scoring.LENSES`; rating thresholds are constants at the top of `risk.py`.
- **The dashboard builds DOM with the `el()` helper.** Never use `innerHTML` with scan data.

## Invariants

These came out of two rounds of outside code review. Each has a test; do not regress them.

- **Risk/reward is in present value, priced at the fill.** `risk.assess` uses the ask to buy and the bid to sell, and discounts the payoff. A fairly priced option must show zero edge at any expiry, and "most you can lose" must equal the cost shown. Display metrics on `ContractAnalysis` (breakeven, `pop`, annualized yield) stay at the mid.
- **A quote below exercise value is rejected**, using plain `max(S-K, 0)` or `max(K-S, 0)` whether or not the feed supplies a volatility. Do not swap in a discounted European floor: it drops legitimate deep in-the-money calls on dividend stocks.
- **Earnings math.** `TRADING_DAY = 1/252`; a report replaces one trading day. `risk.event_variance` uses log returns and takes `(sessions - 1)` ordinary days out of a two-session reaction. `earnings.spans` treats unknown timing as possibly before the open, and `earnings.implied_move` leaves a report-day expiry of unknown timing out of both sides (`_ambiguous`).
- **"No reports" and "cannot tell" are different.** A company that files 6-K, 20-F or 40-F raises `Unsupported` in `EdgarEarnings`, which sets `earnings_unknown`. Never let it look like an ETF, which has no reports at all.
- **The strike window scales with volatility** (`Scanner._strike_window`). A fixed 15% cuts off the deltas the sell lens targets on volatile stocks.
- **One server instance runs for days.** `EarningsService.today` is read each time, never stored at startup. The calendar and the Alpha Vantage block expire and retry; that block is checked and set while holding its lock. `Scanner._lock` allows one scan at a time. `DiskCache.set` is atomic and never fatal. `EdgarEarnings` is paced (`MIN_INTERVAL`). `/` is served with `Cache-Control: no-cache`.
- **Symbols are tidied in `Scanner._scan`** (trim, upper-case, `BRK-B` to `BRK.B`, de-duplicate), so every way in behaves the same.

## Working agreement

- Branch off `main` for each change. Commit, push, open a pull request, merge, and delete the branch only when asked for that step. Merge with a merge commit.
- Do not place trades, and do not say what to trade. Explanations of the numbers are welcome.
- There is no screenshot or OCR path, by decision. Do not add one.
- Keep dependencies minimal, and the dashboard as one vanilla-JS file.
- When an outside review arrives, reproduce its claims against the code before acting on them.

## Not yet verified

Nothing has run against live data. No Alpaca keys have been available, so `AlpacaData` is tested only against response shapes from Alpaca's API reference. Also untested live: the AI brief call, and the SEC ticker-to-CIK file (it needs a contact `User-Agent`). Whether Alpaca's free plan allows the screener endpoints and consolidated price history is unconfirmed; the code falls back and reports a note.

## Next step (proposed, not approved)

Exit timing: an exit plan on every scan result and a premium-selling preset (30 to 45 days, 0.16 to 0.30 delta, skip contracts spanning earnings), then a positions page that checks holdings against exit rules (50% of maximum profit, 21 days to expiry). Ask before building it.

If it is built:

- Keep it in its own module (`app/analysis/exits.py`) with its own dataclasses, not more fields on `ContractAnalysis`.
- Expose the at-the-money volatility term structure, the ex-earnings volatility and the event variance on `UnderlyingContext`. `Scanner._analyze` computes all three and discards them today.
- Count the bid-ask spread twice on an early exit.
- Never present the chance of profit at expiry as the chance of reaching a target before expiry.
- Allow for early assignment around an ex-dividend date on short options. The pricing currently assumes no dividends.

The README roadmap numbers the news layer as phase 2 and paper trading as phase 3. Exit timing has no number there yet.
