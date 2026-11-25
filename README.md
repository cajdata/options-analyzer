# Strike Check

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
   - `ALPHAVANTAGE_API_KEY` (free key, about 25 requests a day): confirmed upcoming dates.
     Its history, which includes EPS surprises, is used only where EDGAR is not set or has
     too few reports, to stay inside the daily limit.
4. Optional: `ANTHROPIC_API_KEY` turns on the AI brief.

```powershell
.venv\Scripts\python.exe -m app serve
```

## Command line

```powershell
.venv\Scripts\python.exe -m app scan                      # discover candidates
.venv\Scripts\python.exe -m app scan AAPL NVDA            # specific symbols
.venv\Scripts\python.exe -m app scan --lens sell --type put --dte-min 20 --dte-max 45
.venv\Scripts\python.exe -m app scan --budget 100         # only contracts that cost $100 or less
.venv\Scripts\python.exe -m app scan --preset sell-premium --exits   # premium-selling setup, with exit plans
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
| Outlay | Cash to open one contract: the ask x 100 to buy; to sell, the strike x 100 for a put or 100 shares for a call |
| Spread | Bid-ask spread as a share of the mid price |
| Breakeven | Stock move needed at expiry for the buyer to break even |
| vs exp. move | Breakeven distance divided by the one-standard-deviation move the options imply by expiry |
| POP | Probability of profit at expiry, for the buyer or the seller depending on the lens |
| Time decay | Theta as a share of the option's price, per day |
| Annualized yield | Premium over collateral (strike for a put, share price for a call), annualized |
| Reward : risk | Average win divided by average loss at expiry, if the stock is as volatile as its own history |
| Rating | Favorable, balanced or unfavorable: expected value beyond +10% or -10% of the premium on that same basis |
| Long shot / tail risk | Label for lopsided payoffs, on the market's own odds: profits less than one time in five, or more than four in five with a rare much larger loss |

**Adds earnings context** for each stock: the next report date (confirmed from the
calendar, or estimated from the usual gap), the close-to-close move on each of the last
twelve reports, and the size of move the options price for the next one. That last
number takes the first expiration after the report and subtracts ordinary volatility,
which comes from the last expiration before the report, else from the next expiration
after it, else from recent realized volatility.

**Applies a budget, if you set one.** Only contracts whose outlay is within the budget
are ranked, and discovery leans toward lower-priced stocks, where contracts in that range
sit closer to the money. A small budget on an expensive stock leaves only contracts far
from the current price; their low chance of profit shows in the numbers beside them.

**Rates risk and reward.** At the market's own prices an option's expected profit is
about zero, so reward-to-risk is just the mirror image of the probability of profit: a
bigger payoff always comes with lower odds. To say something more, the app prices each
contract against the stock's own history instead: recent volatility with earnings days
removed, plus past report-day moves for any report the contract spans. From that it
shows the chance of profit, average win and loss, expected value, the most you can lose
and the loss after a two-standard-deviation move against you. These assume you pay the
ask to buy or receive the bid to sell, so a wide spread counts against a contract, and
they are in today's dollars, so a fairly priced option shows zero at any expiry. Options usually cost more
than history suggests, which is what sellers are paid for carrying the risk of a large
move, so a seller's favorable rating always sits next to those loss figures.

**Draws an exit plan.** A scan scores a contract as held to expiry; the exit plan shows
what happens before then. It reprices the contract on later dates with the stock
unchanged, which isolates what time does to it:

- **Time decay.** What the option is worth in one, two and three weeks, and what closing
  it then would gain or lose. Closing early crosses the bid-ask spread a second time, so
  these use the bid for a holder and the ask for someone buying back.
- **The step across earnings.** A report adds a lump of volatility that vanishes the day
  it lands, so the price drops that day even if the stock does nothing. The plan shows
  the drop, and the stock move that would undo it.
- **A seller's rules of thumb.** The price to buy back at to keep half the premium, when
  decay alone gets there, and the date 21 days before expiry that sellers treat as a time
  stop.

A date on that path is where time alone takes the price. It is not the chance of getting
there, and the rules are for managing risk, not signals.

**Offers a premium-selling preset.** One click, or `--preset sell-premium`, sets the sell
lens to 30 to 45 days, 0.16 to 0.30 delta, and skips contracts still open over an
earnings report. Those filters are also available on their own.

Skipping earnings is strict. A stock with no known report date is left out, an estimated
date is given a week of room in front of it, and the scan says which stocks it left out
and why. It needs an earnings data source to work at all.

**Ranks through a lens.** `buy` ranks long calls and puts; `sell` ranks cash-secured
puts and covered calls. The score is a weighted sum of five components (liquidity,
breakeven or probability, how volatility is priced, decay or yield, delta). "How
volatility is priced" uses the same history-based comparison as the rating, so the
rating is shown beside the score, not added into it a second time. Every weight and
threshold is in `app/analysis/scoring.py`; the rating's are in `app/analysis/risk.py`.

## Limits worth knowing

- **Free Alpaca option quotes are "indicative"**: quotes are modified and trades are
  15 minutes delayed. Screen with them, then check the live quote at your broker.
  Real-time quotes need Alpaca's paid plan (`ALPACA_OPTIONS_FEED=opra`).
- **Free stock quotes come from the IEX exchange only**, a small share of volume. After
  4 p.m. the price shown is an after-hours trade, while every option quote is from the
  close, so an evening scan of a stock that moved after hours (an earnings night, most
  of all) shows option prices that no longer fit the stock.
- **A quote from an earlier day is left out**, and the scan says how many. When a special
  dividend or a split changes a stock's strikes, the contracts that stopped existing go
  on reporting their last quote. A stock whose options have not been quoted at all for
  more than four days (halted or delisted) is left out whole.
- **Open interest is a day or two old** and is blank for a contract listed since
  Alpaca's last update. Volume is the latest session's.
- **A buyout breaks the rating.** Once a cash takeover is agreed, the stock sits at the
  deal price and its options price almost no movement. Next to a volatile history every
  option then looks cheap and the rating says "favorable". Be wary of a stock that jumped
  and whose implied volatility is far below its realized volatility.
- **Pricing math is Black-Scholes** with no dividends and a flat rate (`RISK_FREE_RATE`).
  US equity options are American, so deep in-the-money puts and calls just before a
  dividend are approximate.
- **Risk and reward assumes history repeats** and takes no view on direction. It is a
  comparison of price with past behavior, and past volatility is a rough guide at best.
- **A rating cannot see a report it has no date for.** When no earnings date is known for
  a stock, the scan says so, because options look expensive right before a report for good
  reason. That covers no source being set, a failed lookup, and a foreign company that
  reports on Form 6-K when SEC EDGAR is the only source (EDGAR dates US filers only; add
  an Alpha Vantage key to cover the rest).
- **The exit plan assumes the stock stands still** and that volatility behaves as the
  options currently price it. It does not model dividends, so it cannot see the early
  assignment risk a short call carries just before a stock goes ex-dividend, and it does
  not know market holidays.
- **Earnings dates from SEC filings** take their timing from when the 8-K was accepted.
  A company that files hours after its press release can be off by one session.
- **The AI brief knows only the scan.** It has no news and no view of your account. It
  calls `claude-opus-5-5` (override with `AI_MODEL`) and is billed to your Anthropic key.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Roadmap

1. **Done:** read-only scanner with earnings context, discovery, AI brief, exit plans and
   a premium-selling preset.
2. News layer: classify headlines and filings by event type, and study what followed
   similar events.
3. A positions page that checks what you hold against the exit rules, then paper trading
   on Alpaca with a journal that records which signals worked.
4. Live trading with a human approval on every order.
