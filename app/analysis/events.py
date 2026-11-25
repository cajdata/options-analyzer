"""News context: what kind of event a headline or filing is, and what followed ones like it.

The type comes from fixed, visible rules: a table of keywords for a headline
and the SEC's own item codes for an 8-K. Nothing here reads the story, and
nothing judges whether the news is good or bad. The study then looks back at
the stock's earlier events of the same type and measures how far the stock
moved, next to how far it moves on an ordinary day.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time as clock, timedelta
from functools import lru_cache
from typing import Iterable
from zoneinfo import ZoneInfo

from app.analysis.earnings import window
from app.models import Bar, EventMove, EventStudy, NewsItem

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = clock(16, 0)

RECENT_DAYS = 14  # how far back "recent" news reaches
MAX_RECENT = 10  # items listed for one stock
FOLLOW_SESSIONS = 5  # how long after an event "what followed" looks
MIN_EVENTS = 3  # below this, a study's averages are not worth comparing
MAX_MOVES = 8  # past events listed in a study
# The stock's usual move is measured from the first event on, but over at least this
# many sessions, so two events last week are not compared with a handful of days.
BASELINE_SESSIONS = 60

# Event types, each with the label a person sees.
EVENT_TYPES: dict[str, str] = {
    "earnings": "Earnings results",
    "guidance": "Guidance or outlook",
    "analyst": "Analyst rating or price target",
    "takeover": "Takeover bid or sale of the company",
    "acquisition": "Acquisition or sale of a business",
    "split": "Stock split or special dividend",
    "financing": "Stock or debt offering",
    "payout": "Dividend or buyback",
    "insider": "Insider or large-holder trade",
    "distress": "Bankruptcy, delisting or restatement",
    "incident": "Security incident or outage",
    "drug": "Drug trial or approval",
    "legal": "Legal, regulatory or safety",
    "restructuring": "Restructuring, layoffs or write-down",
    "governance": "Shareholder vote or bylaws",
    "management": "Management or board change",
    "agreement": "Partnership, contract or order",
    "product": "Product or business update",
    "disclosure": "Other company disclosure",
    "calendar": "Scheduled call or presentation",
    "other": "Not classified",
}
# Not studied: earnings have their own section built from report dates, a notice of
# a date is not the event, and unclassified items have nothing in common.
NOT_STUDIED = frozenset({"earnings", "calendar", "other"})
# Types that tend to move a stock in one jump. A recent one is pointed out in the
# list of results, so it is seen before a contract is opened.
FLAGGED = frozenset({"guidance", "takeover", "split", "financing", "distress", "drug"})
# Where the scan's own numbers are known to mislead after an event of this type.
CAUTIONS: dict[str, str] = {
    "takeover": (
        "If this company is being bought for cash, its stock sits at the deal price and its options "
        "price almost no movement, so every option looks cheap next to history and the rating means little."
    ),
    "split": (
        "A split or special dividend changes the strikes and what a contract delivers on the day it "
        "takes effect, and quotes for the old strikes can linger."
    ),
}


def _rule(*alternatives: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alternatives) + ")", re.IGNORECASE)


_ROLES = (
    r"ceo|cfo|coo|cto|cio|chief \w+ officer|chief executive|chairman|chairwoman|president"
    r"|directors?|board member|general counsel|treasurer"
)
_MOVES = (
    r"appoint\w*|nam(?:es?|ed|ing)|elect(?:s|ed)?|hir(?:es?|ed)|promot(?:es?|ed)|reinstat(?:es?|ed)"
    r"|oust(?:s|ed)?|fir(?:es?|ed)|replac(?:es?|ed|ing)|succe(?:eds?|ssion|ssor)|resign\w*|retir\w+"
    r"|step(?:s|ping)? down|depart\w*|transition"
)
_RATINGS = r"buy|sell|hold|neutral|outperform|underperform|overweight|underweight|perform|weight|accumulate|reduce"

# Headline rules, checked in this order: the first that matches decides the type.
# The order matters. "Raises price target" is an analyst note and not guidance, and
# "announces webcast for third quarter earnings" is a date, not the results.
HEADLINE_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "calendar",
        _rule(
            r"webcast",
            r"conference call",
            r"earnings call",
            r"earnings (?:release )?date",
            r"to (?:report|present|participate|host|speak|showcase|attend|exhibit)\b",
            r"investor (?:day|conference|event|meeting)",
            r"fireside",
            r"sets? (?:the )?date",
            r"schedules?\b",
            r"presentations? (?:at|for)\b",
            r"annual meeting",
        ),
    ),
    (
        "analyst",
        _rule(
            r"price targets?",
            rf"(?:up|down)grade[sd]?\b.*\b(?:{_RATINGS}|rating|positive|negative)\b",
            r"initiates? coverage",
            r"coverage (?:of|on)\b",
            rf"(?:{_RATINGS}|equal[- ]weight) rating",
            r"rating changes?",
            r"(?:reiterates?|maintains?|reaffirms?|assigns?|keeps?) .*\brating",
            r"analysts? (?:adjust|rais|cut|lower|rating|see|expect)\w*",
        ),
    ),
    (
        "earnings",
        _rule(
            r"(?:reports?|posts?|announces?|releases?|delivers?|records?)\b.*"
            r"\b(?:first|second|third|fourth|q[1-4]|quarter(?:ly)?|full[- ]year|fiscal|annual)\b.*"
            r"\b(?:results|earnings|revenue|profit|loss|sales|income)\b",
            r"(?:reports?|posts?) (?:\S+ ){0,3}(?:loss|profit|results|earnings|revenue|income)\b",
            r"(?:earnings|eps|revenue|profit|sales) (?:beats?|miss(?:es)?|tops?|falls? short|exceeds?)\b",
            r"(?:beats?|miss(?:es)?|tops?|exceeds?) .*\b(?:estimates|expectations|consensus)\b",
            r"(?:first|second|third|fourth)[- ]quarter (?:\d{4} )?(?:results|earnings|profit|loss|revenue)",
            r"q[1-4] (?:\d{4} )?(?:results|earnings|profit|loss|revenue)",
        ),
    ),
    (
        "guidance",
        _rule(
            # The company's own numbers. A market study it publishes ("2026 Commercial
            # Market Outlook") and a remark from its chief executive are not guidance.
            r"guidance",
            r"(?:financial|revenue|sales|earnings|profit|full[- ]year|fiscal|annual|quarter\w*) (?:outlook|forecast)",
            r"preliminary (?:\w+ )?(?:results|revenue|sales)",
            r"pre-?announces?",
            r"profit warning",
            r"warns? (?:of|on) .*\b(?:revenue|sales|profit|earnings|margins?|demand)\b",
            r"(?:raises?|lifts?|cuts?|lowers?|boosts?|slashes?|reaffirms?|withdraws?|narrows?|trims?|hikes?) .*"
            r"\b(?:outlook|forecast|targets?|view|estimates?|projections?)\b",
            r"expects? .*\b(?:revenue|sales|profit|earnings|margins?)\b",
        ),
    ),
    (
        "takeover",
        _rule(
            r"to be (?:acquired|bought|taken private|sold)\b",
            r"agrees? to be\b",
            r"take[- ]private",
            r"go(?:es|ing)? private",
            r"buyout",
            r"takeover",
            r"tender offer",
            r"(?:acquisition|merger|purchase) (?:offer|proposal|bid)",
            r"unsolicited",
            r"hostile bid",
            r"strategic (?:alternatives|review)",
            r"explor\w+ (?:a |potential )?sale",
            r"sale of the company",
            r"merger agreement",
            r"agrees? to merge",
            r"to (?:merge|combine) with",
            r"all (?:of the )?outstanding shares",
        ),
    ),
    (
        "split",
        _rule(r"(?:stock|share) split", r"reverse (?:stock )?split", r"\d+-for-\d+", r"special (?:cash )?dividend"),
    ),
    (
        # The company is the lender or the investor here, not the one raising money.
        "agreement",
        _rule(
            r"(?:leads?|to lead|provides?|participates? in|underwrit\w+|to underwrite|arranges?|backs?|invests? in)\b.*"
            r"\b(?:financing|credit facility|funding round|loan|series [a-f])\b",
        ),
    ),
    (
        "financing",
        _rule(
            r"(?:public|secondary|registered direct|follow-on|notes?|debt|equity|bond|stock|share|unit) offering",
            r"offering of\b",
            r"prices? (?:\$|its\b|an?\b|upsized|offering)",
            r"private placement",
            r"at-the-market",
            r"convertible",
            r"senior (?:unsecured |secured )?notes",
            r"credit (?:facility|agreement|line)",
            r"term loan",
            r"revolv(?:er|ing)\b",
            r"raises? \$",
            r"funding round",
            r"financing",
            r"shelf registration",
            r"ipo\b",
            r"capital raise",
        ),
    ),
    ("payout", _rule(r"dividends?\b", r"buybacks?\b", r"repurchas\w+")),
    (
        "insider",
        _rule(
            r"insiders?\b",
            r"form 144\b",
            r"form 4\b",
            r"(?:sells?|sold|buys?|bought|purchases?|acquires?|disposes?) .*\bshares\b",
            r"share (?:sale|purchase)\b",
            r"sale of (?:\S+ ){0,3}shares\b",
            r"stake\b",
            r"holdings? (?:disclosure|in|of)\b",
            r"13[dfg]\b",
            r"beneficial owner\w*",
        ),
    ),
    (
        "distress",
        _rule(
            r"bankruptcy",
            r"chapter (?:7|11)\b",
            r"going concern",
            r"delist\w*\b(?!.*\bnotes\b)",  # a bond coming off an exchange is routine
            r"restatement",
            r"restat\w+ .*\b(?:results|financial|earnings|statements)\b",
            r"non-?compliance",
            r"default(?:s|ed)?\b",
            r"deficiency (?:notice|letter)",
            r"material weakness",
            r"receivership",
            r"insolven\w+",
            r"auditor (?:resign|chang|dismiss)\w*",
            r"short[- ]seller report",
        ),
    ),
    (
        "incident",
        _rule(
            r"data breach",
            r"cyber ?attack",
            r"cybersecurity incident",
            r"ransomware",
            r"hack(?:ed|ers?|ing)?\b",
            r"security (?:breach|incident)",
            r"outage",
        ),
    ),
    (
        "drug",
        _rule(
            r"fda\b",
            r"ema\b",
            r"chmp\b",
            r"phase (?:[123]|i{1,3})\b",
            r"clinical",
            r"top-?line",
            r"pdufa",
            r"(?:regulatory|marketing|accelerated) (?:approval|authorization)",
            r"breakthrough therapy",
            r"fast track",
            r"orphan drug",
            r"complete response letter",
            r"label expansion",
            r"(?:trial|study) (?:results|data|meets|shows|fails|misses)\b",
            r"(?:primary|secondary) endpoint",
            r"positive (?:\S+ ){0,2}(?:data|results)\b",
            r"efficacy",
        ),
    ),
    (
        "legal",
        _rule(
            r"lawsuits?\b",
            r"sue[sd]?\b",
            r"court\b",
            r"judge\b",
            r"jury\b",
            r"trial\b",
            r"verdict",
            r"settle(?:s|d|ment)?\b",
            r"litigation",
            r"class action",
            r"investigat\w+",
            r"probes?\b",
            r"subpoena\w*",
            r"antitrust",
            r"doj\b",
            r"ftc\b",
            r"sec (?:charges|investigat\w+|probe|settle\w*)",
            r"fine[sd]\b",
            r"penalt\w+",
            r"sanction\w*",
            r"recall(?:s|ed|ing)?\b",
            r"ban(?:s|ned)?\b",
            r"regulator\w*",
            r"scrutiny",
            r"senat\w+",
            r"congressional",
            r"lawmakers?\b",
            r"testif\w+",
            r"testimony",
            r"indict\w+",
            r"fraud",
            r"violat\w+",
            r"complaint",
            r"injunction",
            r"airworthiness",
            r"faa\b",
            r"nhtsa\b",
            r"tariffs?\b",
            r"export (?:controls?|restrictions?|ban|curbs?|licen[cs]es?)",
            r"whistleblower",
            r"allegations?\b",
            r"accused\b",
        ),
    ),
    (
        "restructuring",
        _rule(
            r"layoffs?\b",
            r"lay(?:s|ing)? off\b",
            r"job cuts?\b",
            r"cuts? .*\b(?:jobs|workforce|staff|employees|positions)\b",
            r"workforce reduction",
            r"restructur\w+",
            r"reorgani[sz]\w+",
            r"impairment",
            r"write-?(?:down|off)s?\b",
            r"(?:plant|factory|store) clos\w+",
            r"clos(?:es|ing|ure of) .*\b(?:plants?|factor(?:y|ies)|stores?|facilit(?:y|ies)|offices?)\b",
            r"shut(?:s|ting)? down\b",
            r"cost[- ]cutting",
            r"wind(?:s|ing)? down\b",
        ),
    ),
    (
        "governance",
        _rule(
            r"(?:share|stock)holders? (?:vote|approve|reject|back|elect)\w*",
            r"proxy\b",
            r"bylaws?\b",
            r"activist (?:investor|fund|shareholder|stake|campaign)",
            r"say[- ]on[- ]pay",
        ),
    ),
    (
        "management",
        _rule(
            rf"(?:{_ROLES})\b.*\b(?:{_MOVES})",
            rf"(?:{_MOVES})\b.*\b(?:{_ROLES})\b",
            r"steps? down\b",
            r"resign(?:s|ed|ation)\b",
            r"departs?\b",
            r"(?:ceo|leadership|management|executive) (?:transition|change|shake-?up|succession)",
            r"to the board\b",
        ),
    ),
    (
        "acquisition",
        _rule(
            r"acquir\w+",
            r"acquisitions?\b",
            r"mergers?\b",
            r"divest\w*",
            r"spin[- ]?offs?\b",
            r"spins? off\b",
            r"(?:sells?|sale of|to sell|selling) .*\b(?:unit|business|division|subsidiary|assets?|operations|brand)\b",
            r"to buy\b",
            r"buys\b",
            r"bought\b",
            r"purchase of\b",
            r"carve-?out",
        ),
    ),
    (
        "agreement",
        _rule(
            r"partner(?:s|ship|ships|ed|ing)?\b",
            r"collaborat\w+",
            r"alliance",
            r"agreements?\b",
            r"agrees? to\b",
            r"contracts?\b",
            r"orders?\b",
            r"ordered\b",
            r"award(?:s|ed)?\b",
            r"select(?:s|ed)\b",
            r"chosen\b",
            r"deal\b",
            r"sign(?:s|ed)\b",
            r"joint venture",
            r"licens\w+",
            r"teams? up\b",
            r"integrat\w+",
            r"adopt(?:s|ed)\b",
            r"joins?\b",
            r"secures?\b",
            r"wins?\b",
            r"commits? to (?:buy|purchase)\b",
            r"memorandum",
            r"utiliz\w+",
            r"powered by\b",
            r"program of record",
            r"lease\b",
        ),
    ),
    (
        "product",
        _rule(
            r"launch(?:es|ed|ing)?\b",
            r"unveil(?:s|ed)?\b",
            r"introduc(?:es|ed|ing)\b",
            r"debuts?\b",
            r"releases?\b",
            r"rolls? out\b",
            r"rollout",
            r"announces? (?:new|the|its)\b",
            r"expands?\b",
            r"opens?\b",
            r"ships?\b",
            r"begins? (?:production|shipping|deliveries|construction)\b",
            r"availab\w+",
            r"upgrad(?:es?|ed)\b",
            r"deliver(?:s|ies|ed)\b",
            r"advances?\b",
            r"production\b",
            r"discontinu\w+",
        ),
    ),
]

# 8-K item codes, most telling first: the first item in this table that a filing
# carries decides its type. Item 9.01 (exhibits) is paperwork and is left out.
FILING_ITEMS: dict[str, tuple[str, str]] = {
    "2.02": ("earnings", "Results of operations"),
    "1.03": ("distress", "Bankruptcy or receivership"),
    "4.02": ("distress", "Earlier financial statements can no longer be relied on"),
    "3.01": ("distress", "Delisting notice, or a listing rule not met"),
    "2.04": ("distress", "An event that speeds up a debt coming due"),
    "4.01": ("distress", "Change of auditor"),
    "5.01": ("takeover", "Change in control of the company"),
    "1.05": ("incident", "Material cybersecurity incident"),
    "2.01": ("acquisition", "Completed acquisition or sale of assets"),
    "2.05": ("restructuring", "Costs of an exit or disposal"),
    "2.06": ("restructuring", "Material impairment"),
    "2.03": ("financing", "New debt or other financial obligation"),
    "3.02": ("financing", "Unregistered sale of shares"),
    "1.01": ("agreement", "Entry into a material agreement"),
    "1.02": ("agreement", "Termination of a material agreement"),
    "5.02": ("management", "Departure or appointment of directors or officers"),
    "5.07": ("governance", "Shareholder vote"),
    "5.03": ("governance", "Change to the charter, bylaws or fiscal year"),
    "3.03": ("governance", "Change to the rights of security holders"),
    "7.01": ("disclosure", "Regulation FD disclosure"),
    "8.01": ("disclosure", "Other events"),
}


# Words that put the company named next in the position of the one being bought.
_BUYING = (
    r"to acquire|acquires?|acquiring|to buy|buys|to purchase|to take over|takeover of|buyout of|acquisition of"
    r"|purchase of|bid for|offer for|offer to buy|deal for|deal to buy|to merge with|merger with|to take private"
)
# First words too common to stand for one company ("General", "American").
_COMMON = frozenset(
    "advanced american applied bank china digital eastern energy first general global great international "
    "national new northern pacific royal southern the united us western".split()
)
_LEGAL_ENDING = re.compile(r"[\s,]+(?:inc|corp|corporation|co|company|ltd|limited|plc|llc|lp|sa|nv|ag|se)\.?$", re.IGNORECASE)


def aliases(symbol: str, name: str | None = None) -> tuple[str, ...]:
    """What a headline may call a company: its ticker, and the first word of its name.

    An entry in capitals is a ticker and has to match exactly, so COST is not
    "cost". Anything else matches in any case. A one- or two-letter ticker is
    left out, and a name that opens with a common word keeps its second word
    too ("general motors").
    """
    out = [symbol.upper()] if len(symbol) >= 3 and symbol.isalpha() else []
    words = re.findall(r"[a-z0-9][a-z0-9'-]*", _LEGAL_ENDING.sub("", (name or "").split("/")[0].strip()).lower())
    if words:
        short = " ".join(words[:2] if words[0] in _COMMON else words[:1])
        if len(short) >= 3:
            out.append(short)
    return tuple(out)


@lru_cache(maxsize=512)
def _bought(names: tuple[str, ...]) -> re.Pattern[str] | None:
    """Matches a headline in which one of `names` is the company being bought."""
    if not names:
        return None
    who = "|".join(re.escape(n) if n.isupper() else f"(?i:{re.escape(n)})" for n in names)
    return re.compile(rf"\b(?i:{_BUYING})\s+(?i:all of\s+|the\s+)?(?:{who})\b")


def headline_type(text: str, names: tuple[str, ...] = ()) -> str:
    """A headline's event type. `names` are what the company may be called (see `aliases`).

    The feed tags the buyer, its bankers and the company being bought with the
    same headline, and the keyword rules cannot tell them apart. With the
    company's own name to go on, "Sanofi to Acquire Insmed" is a takeover for
    Insmed and an acquisition for everyone else, which matters because a
    buyout is the case where the scan's rating stops meaning anything.
    """
    bought = _bought(names)
    if bought is not None and bought.search(text):
        return "takeover"
    for kind, pattern in HEADLINE_RULES:
        if pattern.search(text):
            return kind
    return "other"


def classify(item: NewsItem, names: tuple[str, ...] = ()) -> NewsItem:
    """The item with its event type set. A filing also gets a title that spells out its items."""
    if item.kind != "filing":
        return replace(item, event=headline_type(item.title, names))
    carried = [FILING_ITEMS[code] for code in FILING_ITEMS if code in item.items]
    if not carried:
        return replace(item, event="other")
    form = item.title.split(":", 1)[0]
    return replace(item, event=carried[0][0], title=f"{form}: " + "; ".join(what for _, what in carried))


def _felt(when: datetime) -> tuple[date, str]:
    """The New York day an item landed on, and whether that was after the close."""
    local = when.astimezone(ET)
    return local.date(), "post" if local.time() >= MARKET_CLOSE else "pre"


def _mean_abs(values: Iterable[float]) -> float | None:
    sizes = [abs(v) for v in values]
    return statistics.fmean(sizes) if sizes else None


def study(items: list[NewsItem], bars: list[Bar], earnings_days: set[date], kinds: Iterable[str]) -> dict[str, EventStudy]:
    """What followed this stock's past events of each type in `kinds`.

    Each event is placed on the first session whose close reflects it: the same
    day for news before the close, the next trading day otherwise. Several
    items of one type on one session are one event. A session that also carries
    an earnings reaction is left out, since that move belongs to the report.

    Beside the average move on those sessions goes the average move on any
    session over the same stretch, earnings days left out, because a 3% day
    means little for a stock that often moves 3%. The same is done for the
    `FOLLOW_SESSIONS` sessions after each event, against every span of that
    length in the stretch.

    "Any session" includes the event days themselves. Leaving them out would
    make a cleaner contrast for a rare event, but for a company in the news
    most days (seen live: 114 of 145 sessions) it would leave a few odd days to
    stand for the ordinary one.
    """
    days = [b.day for b in bars]
    closes = [b.close for b in bars]
    out: dict[str, EventStudy] = {}
    for kind in dict.fromkeys(kinds):
        if kind in NOT_STUDIED:
            continue
        sessions: dict[int, tuple[int, NewsItem]] = {}  # reaction session: (session before, earliest item)
        for item in sorted((i for i in items if i.event == kind), key=lambda i: i.when):
            found = window(*_felt(item.when), days)
            if found is None or days[found[1]] in earnings_days:
                continue
            sessions.setdefault(found[1], (found[0], item))
        if not sessions:
            continue

        moves = []
        for after, (before, item) in sorted(sessions.items(), reverse=True):
            later = after + FOLLOW_SESSIONS
            moves.append(
                EventMove(
                    day=days[after],
                    title=item.title,
                    move_pct=closes[after] / closes[before] - 1.0,
                    after_pct=closes[later] / closes[after] - 1.0 if later < len(closes) else None,
                )
            )
        start = max(1, min(min(sessions), len(days) - BASELINE_SESSIONS))
        ordinary = _mean_abs(
            closes[i] / closes[i - 1] - 1.0 for i in range(start, len(days)) if days[i] not in earnings_days
        )
        followed = [m.after_pct for m in moves if m.after_pct is not None]
        spans = range(start, len(days) - FOLLOW_SESSIONS)
        out[kind] = EventStudy(
            event=kind,
            label=EVENT_TYPES[kind],
            count=len(moves),
            since=moves[-1].day,
            mean_abs=_mean_abs(m.move_pct for m in moves),
            up=sum(1 for m in moves if m.move_pct > 0),
            down=sum(1 for m in moves if m.move_pct < 0),
            ordinary_abs=ordinary,
            after_count=len(followed),
            mean_abs_after=_mean_abs(followed),
            ordinary_abs_after=_mean_abs(closes[i + FOLLOW_SESSIONS] / closes[i] - 1.0 for i in spans) if followed else None,
            moves=moves[:MAX_MOVES],
        )
    return out


def context(
    items: list[NewsItem], bars: list[Bar], earnings_days: set[date], today: date, names: tuple[str, ...] = ()
) -> tuple[list[NewsItem], dict[str, int], dict[str, EventStudy]]:
    """One stock's news for a scan: the latest items, a count of recent ones by type, and the studies.

    `items` are everything on record for the stock, classified or not, and
    `names` are what a headline may call the company. A study is built for
    each type seen in the last `RECENT_DAYS` days, from the whole record.
    """
    classified = sorted((classify(i, names) for i in items), key=lambda i: i.when, reverse=True)
    cutoff = today - timedelta(days=RECENT_DAYS)
    recent = [i for i in classified if _felt(i.when)[0] >= cutoff]
    counts = Counter(i.event for i in recent)
    order = [kind for kind in EVENT_TYPES if kind in counts]
    return recent[:MAX_RECENT], {kind: counts[kind] for kind in order}, study(classified, bars, earnings_days, order)
