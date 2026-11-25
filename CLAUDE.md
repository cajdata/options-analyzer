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
- `.claude/launch.json` defines two dashboards for the preview tools: `strike-check-demo` on port 8765 (synthetic data) and `strike-check-live` on port 8766 (needs the keys in `.env`). The server does not reload: restart it after changing Python code.
- With `ANTHROPIC_API_KEY` set, the live dashboard shows a "Write brief" button and `scan --brief` works. Each use is a billed call: ask before making one.

## Architecture

One scan flows in a straight line:

1. **Providers** (`app/providers/`) fetch data. `MarketData` is the protocol; `AlpacaData` is the real one and `DemoMarket` the synthetic one. `EarningsService` combines earnings sources (SEC EDGAR, Alpha Vantage), caches them on disk, and estimates a missing date.
2. **`Scanner`** (`app/scanner.py`) picks symbols (given ones, or discovery from the watchlist, screeners and `data/universe.txt`), then analyzes each on a small thread pool.
3. **Analysis** (`app/analysis/`) runs in a fixed order per symbol: `contracts` (per-contract metrics) → `earnings` (past reactions, implied move) → `risk.history_vol` (history-based volatility) → `scoring.score` → `risk.assess` (risk/reward). `scoring.rank` runs last, across all symbols. `exits.build` then draws an exit plan for each ranked contract.
4. **Surfaces** read the resulting `ScanResult`: `app/server.py` (FastAPI plus `public/index.html`), `app/__main__.py` (command line), `app/ai.py` (optional Claude write-up).

Things that are not obvious from any one file:

- **The lens decides late.** `contracts.analyze_contract` produces lens-independent metrics priced at the mid. Only afterwards, in `Scanner._analyze`, does the lens ("buy" or "sell") set `pop`, `outlay`, `score` and `risk_reward`. A `ContractAnalysis` is therefore filled in across several modules.
- **`history_vol` is the shared yardstick.** `risk.history_vol` blends ex-earnings realized volatility with past report-day variance. The score's volatility component and the risk/reward rating both start from it, which is why the rating is shown beside the score and never added into it.
- **Provider caveats travel as notes.** A provider appends to `market.notes`; the scanner drains them into `ScanResult.warnings`, which every surface displays. Use this for anything the person should know (a feed fallback, missing open interest).
- **Time is injected.** `Scanner(today=)`, `EarningsService(today=, clock=)`, `AlpacaData(now=, sleep=)` and `EdgarEarnings(sleep=)` exist so tests never depend on the wall clock. Tests pin `TODAY = date(2026, 11, 24)`. `--demo` and the demo dashboard are the exception: `build_scanner` starts their market from the wall clock, so `tests/test_scanner.py` holds it at `TODAY` with the `demo_day_is_pinned` fixture. Without it the command-line tests pass only on that one day.
- **Tunable numbers live in two places.** Score weights and thresholds are in `scoring.LENSES`; rating thresholds are constants at the top of `risk.py`.
- **Shared data types live in `app/models.py`**, including the exit-plan dataclasses, so they can be imported anywhere without a cycle. `exits.py` holds only the logic. Exit plans are kept in `ScanResult.exit_plans`, keyed by contract symbol, and not on `ContractAnalysis`.
- **Presets are data.** `scanner.PRESETS` maps a name to `ScanRequest` fields. The API serves it, the dashboard builds its buttons from it, and the command line reads it, so a preset is defined once.
- **The dashboard builds DOM with the `el()` helper.** Never use `innerHTML` with scan data.

## Invariants

These came out of two rounds of outside code review. Each has a test; do not regress them.

- **Risk/reward is in present value, priced at the fill.** `risk.assess` uses the ask to buy and the bid to sell, and discounts the payoff. A fairly priced option must show zero edge at any expiry, and a buyer's "most you can lose" must equal the cost shown. (A seller's outlay is collateral, a different number: the most a short put can lose is the strike less the bid, and a short call has no cap on its own.) Display metrics on `ContractAnalysis` (breakeven, `pop`, annualized yield) stay at the mid.
- **A quote below exercise value is rejected**, using plain `max(S-K, 0)` or `max(K-S, 0)` whether or not the feed supplies a volatility. Do not swap in a discounted European floor: it drops legitimate deep in-the-money calls on dividend stocks.
- **Earnings math.** `TRADING_DAY = 1/252`; a report replaces one trading day. `risk.event_variance` uses log returns and takes `(sessions - 1)` ordinary days out of a two-session reaction. `earnings.spans` treats unknown timing as possibly before the open, and `earnings.implied_move` leaves a report-day expiry of unknown timing out of both sides (`_ambiguous`).
- **"No reports" and "cannot tell" are different.** A company that files 6-K, 20-F or 40-F raises `Unsupported` in `EdgarEarnings`, which sets `earnings_unknown`. Never let it look like an ETF, which has no reports at all.
- **The strike window scales with volatility** (`Scanner._strike_window`). A fixed 15% cuts off the deltas the sell lens targets on volatile stocks.
- **One server instance runs for days.** `EarningsService.today` is read each time, never stored at startup. The calendar and the Alpha Vantage block expire and retry; that block is checked and set while holding its lock. `Scanner._lock` allows one scan at a time. `DiskCache.set` is atomic and never fatal. `EdgarEarnings` is paced (`MIN_INTERVAL`). `/` is served with `Cache-Control: no-cache`.
- **An exit plan is a standing-still path, and says so.** `exits.build` reprices a contract on later dates with the stock unchanged. The path starts at the quoted mid and is never below exercise value. Closing early uses the bid for a holder and the ask to buy back, so the spread is counted twice. An earnings report's variance leaves on `earnings.reaction_day`, and the last chance to close before it is `earnings.session_before` that day, never a weekend. A date on the path must never be shown as the chance of getting there, and the profit target and time stop (`exits.PROFIT_TARGET`, `exits.TIME_STOP_DTE`) are presented as rules of thumb.
- **The exit path counts time in calendar days, so the report takes one calendar day** (`exits.REPORT_SLOT`), not `TRADING_DAY`. With the longer trading-day slot, stepping past the report hands back half a day of ordinary time, and at one or two days to expiry the option came out priced higher after the report than before. The one-off decompositions in `risk.py` and `earnings.py` keep `TRADING_DAY`. The path must start at the quoted mid and never rise with the stock unchanged, at any expiry.
- **A reading of zero is a reading.** When the options price no extra move for a report, the exit plan shows no step. Past reports size the step only when the options give no reading at all, so the plan never contradicts the earnings section beside it.
- **Skip-earnings fails closed.** With `avoid_earnings` on, a stock whose report date is unknown is left out, an estimated date is given `scanner.ESTIMATE_MARGIN` of room, and discovery does not spend a slot on a stock whose every contract would be discarded (nor give it the usual bonus for reporting soon). Each case adds a warning saying which stocks were left out and why.
- **`earnings_unknown` means no date at all.** A date from the calendar is a date, even when the history lookup behind it failed.
- **Symbols are tidied in `Scanner._scan`** (trim, upper-case, `BRK-B` to `BRK.B`, de-duplicate), so every way in behaves the same.

These came out of the first run against live data (2026-11-24). Each has a test built from a captured response (`tests/fixtures/alpaca_live.json`).

- **A quote from an earlier day is not a quote.** Alpaca's snapshot holds the last quote it saw for a contract. When a special dividend or a split changes a stock's strikes, the contracts that stopped existing go on answering with their final quote, a day old or a year old, and they are missing from the active contract list. `AlpacaData.option_chain` leaves out any quote from an earlier day than the newest quote in its chain (`alpaca._session_day`) and adds a note saying how many. The newest quote sets the day, never the clock, so a weekend scan keeps Friday's quotes; a quote stamped before the 9:30 open does not move the day on.
- **A chain nobody has quoted for days is left out whole.** The rule above compares quotes with each other, so it cannot see a chain whose quotes are all equally old (a halted or delisted stock, whose own last trade is just as old). Only here does the clock decide: if the newest quote is more than `alpaca.MAX_QUOTE_AGE_DAYS` (four, room for a three-day weekend) before `AlpacaData._now()`, none is used and a note says so. Tests that build a chain pin that clock.
- **Volume is the latest session's.** A snapshot's daily bar is from the last day the contract traded, which is often not today. Volume from an earlier day's bar is reported as 0.
- **A stock with nothing to price costs nothing more.** `Scanner._analyze` returns as soon as no quote in the chain can be priced, before the open-interest and earnings lookups. The day's movers include warrants and tiny companies with no options, and an earnings lookup for one can spend an Alpha Vantage request. Such a stock is named in the "no two-sided option quotes" note and in no note about ratings.

## What live data looks like

Measured on 2026-11-24 after the close, with paper keys on the free plan. Use these when judging whether a result is a bug.

- **Greeks and implied volatility arrive for most contracts, not all.** The indicative feed sent them for 80% to 97% of a chain. It sends none for a contract with no bid, and none for many deep in-the-money ones. `contracts.analyze_contract` then solves its own, in the same units (theta per day, vega per volatility point).
- **Open interest lags and has gaps.** It comes from the paper trading host's contract list, as text, and was dated 2026-11-20, two trading days before the scan. A contract listed since then has none: 13% to 41% of a chain. A known zero never appears. `scoring._liquidity` scores a contract with no figure on its spread alone.
- **Dropped quotes are far from the money.** After the close, 1% to 12% of a chain was dropped: no bid (far out of the money), a midpoint below exercise value (deep in the money: delta 0.76 or more, above 0.9 for nearly all, where the bid sits well under exercise value and the ask just above it), and a handful with no volatility that solves. None was a contract either lens ranks.
- **After the close the stock keeps trading and the option quotes do not.** The scan price is the last trade on IEX, which after 4 p.m. is an after-hours trade, while every option quote is from the close. ORCL was 1.0% below its close and 44 of its puts had their whole quote under exercise value. On an earnings night that gap is the whole move, and out-of-the-money quotes are just as out of date with no check to catch them. Alpaca's own implied volatility and Greeks move with it: 45 minutes apart, with the option quote unchanged and NVDA 1.2% lower after hours, a put's implied volatility went from 58.8% to 56.5% and a call's from 64.5% to 67.0%. So when a stock falls after the close its puts look cheaper and its calls dearer than they were. Not handled yet: the scan neither uses the close nor says the stock has moved since. Two facts for whoever fixes it: the IEX snapshot's `dailyBar.c` stayed at the 4 p.m. price while `latestTrade` moved on (checked until 5:20 p.m.), and option quotes did not all stop at 4:00 (SPY's ran to 4:15, and some AAPL and NVDA quotes were stamped as late as 4:14), so no single stock price fits a whole chain. Switching to the close is not enough on its own, because Alpaca's implied volatility and Greeks would still belong to the later price.
- **Consolidated price history is allowed** on the free plan when the request ends at least 15 minutes ago, which `daily_bars` already does. The IEX fallback did not trigger.
- **The screeners are allowed** on the free plan. The movers list includes warrants and tiny companies with no options, so a discovery scan loses a few of its 20 slots to them. Each costs one chain request and is named in one note.
- **The stock price is only as fresh as the last IEX trade.** A stock that rarely trades there can be weeks out of date (a warrant on the movers list last traded five weeks earlier). An IEX daily close differs a little from the official one, and `change_pct` is not adjusted for a dividend or a split.
- **Alpha Vantage's calendar is short and slow.** It listed 669 companies for three months and had no date yet for the large companies' January reports, so most upcoming dates are still estimates. One history request in ten hung for the full 30-second timeout. A first discovery scan took 48 seconds, 38 of them waiting on Alpha Vantage; with earnings history cached it takes about 3.
- **Rate limits were never close.** Each Alpaca host allows 200 requests a minute. A discovery scan made 68 requests in all.
- **A buyout breaks the rating.** Once a cash takeover is agreed the stock sits at the deal price and its options price almost no movement (INSM: 6% implied against a volatile history). Every option then looks cheap next to history and the rating says "favorable". The scan does not detect this.

## Working agreement

- Branch off `main` for each change. Commit, push, open a pull request, merge, and delete the branch only when asked for that step. Merge with a merge commit.
- Do not place trades, and do not say what to trade. Explanations of the numbers are welcome.
- There is no screenshot or OCR path, by decision. Do not add one.
- Keep dependencies minimal, and the dashboard as one vanilla-JS file.
- When an outside review arrives, reproduce its claims against the code before acting on them.

## Verified live, and not yet

The first live run was on 2026-11-24, after the 4 p.m. close, with paper keys on Alpaca's free plan. Every endpoint `AlpacaData` uses answered and matched its parser: stock snapshots, daily bars, option chain snapshots (with paging), the option contract list for open interest, most-actives and movers. Also confirmed: the SEC ticker-to-CIK file with a real contact `User-Agent`, SEC earnings history, a foreign filer (TSM) flagged as "earnings date unknown" when SEC EDGAR is the only source, an ETF (SPY) not flagged, `BRK-B` on every endpoint, Alpha Vantage's calendar and history, the command line, the preset and the live dashboard.

The AI brief was called once, through `ai.write_brief` with the app's own settings (`claude-opus-5-5`, effort high, the fallback beta). It answered in 57 seconds with a parsed brief of four picks, all contracts from the scan, and it carried the feed and estimated-date caveats. A 20-contract discovery scan sent 26,119 tokens and got 5,492 back: 21 cents. With the 16,000-token cap on the answer, the most one brief can cost is about 43 cents.

Not yet verified:

- **Anything during market hours.** How many quotes the exercise-value check drops while the stock is moving, how old the indicative quotes are during the session, and how soon after the open every contract has a quote from that day (until it does, the earlier-day rule leaves it out). Each stock that loses contracts that way adds its own note; if a scan soon after the open shows many, fold them into one as was done for stocks with no quotes.
- **Before the open.** Whether the feed sends quotes ahead of 9:30. `alpaca._session_day` ignores them when choosing the day, on the assumption that it can.
- **The morning of a split or special dividend, before the open.** The strikes have changed overnight but the newest quotes are still the day before's, so the contracts that stopped existing are not yet older than anything and are not caught. On 2026-11-24 they were exactly the contracts missing from the active contract list (64 of 64 for COST, 14 of 14 for TQQQ), which `open_interest` already fetches, so that list is the likely remedy. Not done, because it is unknown when Alpaca updates the list and whether a contract listed during the day appears in it at once.
- **The numbers against a broker.** No contract has been compared with Robinhood's quote, Greeks or open interest.
- **The `--brief` flag and the dashboard's brief button themselves.** The one live call went through the function they both use, not through either of them.
- **Thinly traded stocks.** The age of the last IEX trade is not checked, so a stock that rarely trades on IEX may be priced from an old trade.
- **Python 3.11 on live shapes.** Alpaca's nanosecond time stamps parse on 3.13; the captured-response tests have not yet run on 3.11 (CI will).
- **Live keys and the paid OPRA feed.**

## Next step (proposed, not approved)

A positions page: the person enters what they hold, and the app checks each position against the exit rules (profit target reached, time stop reached, earnings in a few days, stock through the strike). Ask before building it.

If it is built:

- It needs a small persistent store outside `.cache/` for entries (contract, side, quantity, entry price and date). The server keeps only the last ten scans in memory.
- Refresh held contracts with Alpaca's snapshot-by-symbols endpoint instead of pulling whole chains.
- Reuse `exits.build` for the forward path; do not write a second repricing model.
- The chance of reaching a target before expiry is path-dependent. If it is ever shown, it needs its own model (first-passage or simulation), never the chance of profit at expiry.
- Early assignment around an ex-dividend date matters for short options and needs a dividend data source. The pricing currently assumes no dividends.
- With free delayed quotes, signals are end-of-day quality, not intraday.

The README roadmap lists this with paper trading as item 3, after the news layer.
