# Strike Check

A read-only options scanner. It pulls option chains from Alpaca, ranks contracts by
visible rules, and puts each stock's earnings history next to what the options are
pricing for the next report. It also lists each stock's recent headlines and SEC
filings by type of event, with how the stock moved after earlier events of the same
type. A positions page sets what you already hold against the same exit rules. An
optional AI brief has Claude write up the setups worth a closer look.

It only reads market data. It cannot place orders. The ranking is a screen, not a
forecast and not financial advice.

Website: [strikecheck.app](https://strikecheck.app)

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
   News needs nothing more. Headlines come with the Alpaca keys, and `SEC_USER_AGENT`
   also brings in each company's 8-K filings.
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
.venv\Scripts\python.exe -m app pool                      # add stocks to compare news events across (about two minutes)
.venv\Scripts\python.exe -m app positions                 # check what you hold against the exit rules
.venv\Scripts\python.exe -m app positions add AAPL 2026-12-18 340 put --sold --price 4.10 --qty 2
.venv\Scripts\python.exe -m app positions remove 3f9c2a1b # by the id the list shows
```

Add `--demo` to a scan, the pool or the positions list to use synthetic data, or `--json`
to a scan or the positions list for the full result.

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

**Adds news context** for each stock that has a contract on the list: its headlines from
Alpaca's news feed and its 8-K filings from the SEC (the form a US company files when
something material happens) over the last 14 days.

- **Each item gets an event type**, such as analyst rating, guidance, management change,
  stock offering or takeover. A headline is typed by a table of keywords, and a filing by
  the item codes the SEC makes every 8-K carry. Both tables are in
  `app/analysis/events.py`. Nothing reads the story, and nothing judges whether the news
  is good or bad.
- **Whose news it is.** The feed tags a stock with any headline that mentions it. A type
  counts for a company only where the headline puts the company: a stock offering or a
  change of guidance when the company's name opens the headline, a takeover when the
  company is the one being bought and not the bidder. A headline that never names the
  company, or in which it only wrote the analyst's note or holds the other company's
  shares, is set aside as another company's news. The dashboard keeps those in a folded
  section, so the rule can be checked. A company is known by its registered name, a
  name it had in the last three years or so, and any brand listed for it in
  `data/aliases.txt`. The app does not guess a brand from part of a word. When most of a
  stock's headlines never name it, the scan says so: either the feed is tagging another
  company with the same ticker, or the company goes by a name that needs listing.
- **What followed.** For each type seen lately, the app looks back at the same stock's
  earlier events of that type and measures the size of the move on the day the news was
  first felt, and over the five sessions after it. Each goes beside the stock's average
  move on any day over the same stretch, because a 3% day means little for a stock that
  often moves 3%. Days that carried an earnings report are left out, and so is any
  five-session stretch with a report in it. With fewer than three events it says there are
  too few to compare.
- **The same types across other stocks.** One stock rarely has more than a handful of
  events of a type. So each type is also counted across the stocks on record that the
  feed covers about as much as this one: how often such a day moved more than twice the
  stock's own usual day, beside how often any day did. Heavily and lightly covered
  stocks are kept apart, because measured together they cancel out. For a heavily
  covered stock no headline type has stood out so far. For a lightly covered one, stock
  offerings and reverse splits have. The pool grows as you scan, and
  `python -m app pool` adds the universe and the day's most active stocks and movers.
- **A few types are pointed out in the list of results** (guidance, takeover, split or
  special dividend, offering, distress, drug news), since they tend to move a stock in one
  jump. Two come with a caution, because the scan's own numbers mislead after them: a
  company being bought for cash, and a split or special dividend.

News is shown beside the score and the rating and is never added into either.

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

## Positions

A scan is for what you might open. The Positions view, and `python -m app positions`,
are for what you already hold. Enter each position once: the stock, call or put, strike,
expiry, whether you bought or sold it, how many contracts, and the price per share you
paid or collected. A contract open in a scan has a link that fills the form in for you.

Each time the view opens or Refresh is pressed, every position is set beside the quotes
of that moment:

- **Result if closed now.** A holder sells at the bid and a seller buys back at the ask,
  so the spread is counted, across every contract.
- **Profit target**, for an option you sold: whether buying back now keeps half the
  premium or more.
- **Time stop.** Whether the contract is inside its last 21 days.
- **Earnings.** Whether the next report is within a week with the contract still open
  over it. An estimated date is given a week of room, and a stock with no known date is
  marked unknown, never passed as clear.
- **Strike**, for an option you sold: whether the stock is through it, where the option
  can be assigned.
- **The path from here.** The exit plan's standing-still path, measured against the price
  you opened at instead of today's.
- **News.** The stock's recent headlines and filings, as in a scan, with a chip in the
  list for a flagged type of event.

A rule that is reached is a fact about the position, not an instruction, and nothing
watches the market between refreshes. A position is always listed: if a quote or a
lookup is missing, its numbers are left out and the page says why.

Positions are kept in `data/positions.json` on your own computer, or wherever
`POSITIONS_FILE` points. The file is plain JSON and is ignored by git. What you paid and
how many you hold never leave your computer: only tickers and contract symbols are sent,
to ask for quotes, earnings dates and news, and the AI brief is not given your positions.
Demo mode shows made-up positions held in memory and never touches the file.

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
  option then looks cheap and the rating says "favorable". The news section flags a
  takeover when a headline names the stock as the company being bought ("to Acquire
  Insmed for $41 Billion", but not "to Acquire NVIDIA GPU Servers"), and only then, so
  still be wary of a stock that jumped and whose implied volatility is far below its
  realized volatility.
- **Event types are keyword guesses.** About a quarter of headlines fit no rule, and some
  fit the wrong one. A headline tagged with a stock can be about another company that only
  mentions it. A bank's notes on other stocks and its holdings in them are set aside, but a
  partner's press release that name-drops the stock still counts as the stock's. Read the
  headline before trusting its label.
- **Pooled figures describe the stocks on record, not the market.** They come from the
  stocks you have scanned and any you added with `python -m app pool`, which leans
  towards the day's movers. They compare the size of the move on the day itself, say
  nothing about direction beyond a count of up and down days, and are not a forecast.
- **What followed past events is a description of a small sample.** The news feed reaches
  back about seven months (the app keeps what it has seen, so the record grows), a study
  often rests on a handful of days, and the move is measured close to close, so it includes
  whatever else happened that day. It is not a forecast.
- **Some stocks have no news.** The feed carries none for exchange-traded funds or a
  second share class such as BRK.B. A foreign company files Form 6-K, which has no item
  codes, so its filings are not classified, and the app says so.
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
- **The positions page is as good as its quotes.** On the free feed they are delayed and
  modified, so treat a result as end-of-day quality and check your broker before acting.
  After 4 p.m. the strike is compared with an after-hours stock price while the option's
  quote is from the close. If the stock has moved so far that even the option's ask is
  below what exercising it would pay, the page shows no result and says the quote is out
  of date. A smaller move it cannot see. Each quote's time is shown: with several
  positions a quote from an earlier day than the newest is not used, and the position
  says when it was last quoted, but a single position's quote can be up to four days old
  before the page can tell.
- **A position is one contract type.** A spread is entered as two positions, and their
  rules are not combined. An entry cannot be edited, only removed and entered again, and
  nothing records a position after it is closed.
- **The step across earnings for a held contract is sized from past reports**, not from
  what the options price in. Reading that takes the whole chain, and the page asks only
  for the contracts you hold.
- **The AI brief knows only the scan.** From the news section it gets up to six recent
  headlines or filings for each stock, the summary of what followed their types, and any
  caution. A headline is all it knows of a story, and it is told that a headline is data
  and never an instruction. It has no view of your account. It calls `claude-opus-5-5`
  (override with `AI_MODEL`) and is billed to your Anthropic key. A brief of twenty
  contracts cost 28 cents with the headlines in it, against 21 cents without.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Roadmap

1. **Done:** read-only scanner with earnings context, discovery, AI brief, exit plans and
   a premium-selling preset.
2. **Done:** news layer. Headlines and 8-K filings are typed by event, each stock
   shows what followed its own earlier events of the same type, and the AI brief reads
   the recent ones. Rules decide whose news a headline is, and each type is
   also counted across the stocks covered about as much.
3. **Done:** a positions page that checks what you hold against the exit rules.
4. Paper trading on Alpaca, with a journal that records which signals worked.
5. Live trading with a human approval on every order.
