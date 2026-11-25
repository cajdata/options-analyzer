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
from typing import Iterable, NamedTuple
from zoneinfo import ZoneInfo

from app.analysis.earnings import window
from app.models import Bar, EventMove, EventStudy, NewsItem, PooledStudy

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = clock(16, 0)

RECENT_DAYS = 14  # how far back "recent" news reaches
MAX_RECENT = 10  # items listed for one stock, before any flagged one a busy week pushed out
FOLLOW_SESSIONS = 5  # how long after an event "what followed" looks
MIN_EVENTS = 3  # below this, a study's averages are not worth comparing
MAX_MOVES = 8  # past events listed in a study
# The stock's usual move is measured from the first event on, but over at least this
# many sessions, so two events last week are not compared with a handful of days.
BASELINE_SESSIONS = 60

# Pooling a type of event across stocks.
BIG_MOVE = 2.0  # a day is big at more than this many times the stock's own usual day
MIN_POOLED = 20  # below this many pooled events nothing is compared
COVERAGE_DAYS = 180  # how far back a stock's headlines are counted to say how much it is covered
HEAVY_HEADLINES = 90  # that many of its own in that time, about one every other day, is heavy coverage
GROUPS: dict[str, str] = {"light": "lightly covered", "heavy": "heavily covered"}
# What a stock adds to a pool for one type of event, in this order.
TALLY = ("events", "big", "up", "down", "sessions", "big_sessions")

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
    "another": "Another company's news",
    "calendar": "Scheduled call or presentation",
    "other": "Not classified",
}
# A headline the feed tags with a stock though the news is someone else's: it never
# names the company, or the company is the one writing the analyst's note or holding
# the shares. Such an item is kept to be looked at, and counts towards nothing.
ELSEWHERE = "another"
MAX_ELSEWHERE = 10  # how many of them are kept for one stock
# Not studied: earnings have their own section built from report dates, a notice of
# a date is not the event, and neither unclassified items nor other companies' news
# have anything in common.
NOT_STUDIED = frozenset({"earnings", "calendar", "other", ELSEWHERE})
# These happen to the company that announces them, so they count for a company only
# when its name opens the headline. "IREN Secures $3.6 Billion Financing for NVIDIA GPU
# Acquisition" is IREN raising money, and "Morgan Stanley Raises Tesla Q2 Delivery
# Estimate" is not Tesla's guidance. Drug news is left out of this on purpose: "FDA
# Approves Eli Lilly's Mounjaro" is the usual wording for the company's own news.
IN_FRONT = frozenset({"guidance", "split", "financing", "distress"})
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
# Words that make "positive data" a medicine's and not a benchmark's.
_CLINICAL = r"trials?|stud(?:y|ies)|patients?|therap\w+|treatment|vaccine|cancer|disease|dos(?:e|ing)|candidate|investigational"

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
        # The company is the lender, the investor or the banker here, not the one raising money.
        "agreement",
        _rule(
            r"(?:leads?|to lead|provides?|participates? in|underwrit\w+|to underwrite|arranges?|backs?|invests? in)\b.*"
            r"\b(?:financing|credit facility|funding round|loan|series [a-f])\b",
            r"(?:selected|chosen|hired|tapped|picked) (?:to|as|for)\b.*\b(?:ipo|offering|listing)\b",
            r"to lead\b.*\b(?:ipo|offering|listing)\b",
            r"working with\b.*\bon\b.*\b(?:ipo|offering|listing)\b",
            r"(?:participat\w+ in|pledges?|commits?|(?:expresses?|indicates?) interest in|pitch(?:es|ing)?)\b.*"
            r"\b(?:ipo|offering|funding round|financing)\b",
        ),
    ),
    (
        "financing",
        _rule(
            r"(?:public|secondary|registered direct|follow-on|notes?|debt|equity|bond|stock|share|unit) offering",
            r"offering of\b",
            # A company prices a product too ("Prices Its New Headset at $3,499"), so the
            # thing priced has to be named as a security.
            r"prices? (?:upsized )?offering",
            r"prices? (?:\$|its\b|an?\b|upsized\b).*\b(?:offering|notes|bonds|debentures|placement)\b",
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
            # A default on something owed. The default search engine on a phone is not one.
            r"defaults? on (?:\S+ ){0,4}?(?:debt|loans?|bonds?|notes|payments?|obligations?|credit|agreement|lease|mortgage)\b",
            r"defaulted\b",
            r"in default\b",
            r"(?:debt|loan|bond|credit|payment|covenant|mortgage) defaults?\b",
            r"(?:event|notice) of default\b",
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
            r"phase (?:[123]|i{1,3})[ab]?\b",
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
            # "Positive benchmark data" for a chip is not a drug, so a clinical word has to be there too.
            rf"positive (?:\S+ ){{0,2}}(?:data|results)\b.*\b(?:{_CLINICAL})\b",
            rf"(?:{_CLINICAL})\b.*\bpositive (?:\S+ ){{0,2}}(?:data|results)\b",
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


# Phrases a company is bought with. After one of these, the company's name is enough.
_TAKING = (
    r"to acquire|acquires?|acquiring|acquisition of|to take over|takeover of|buyout of|to take private"
    r"|to merge with|merger with|bid for|offer for|(?:offer|deal|agrees?|agreement) to buy"
)
# Shares, chips and aircraft are bought with these too, so here the price has to come next.
_PAYING = r"to buy|buys|to purchase|purchase of|deal for"
# How a headline goes on once it has named the company being bought: it ends, or it
# turns to the price and the terms. Anything else after the name is something of the
# company's, not the company ("Nvidia servers", "Pfizer brands", "Insmed's lung unit").
_THEN = r"(?=\s*$|\s*[,.;:!?()]|\s+[-–—]|\s+(?:for|in|at|to|by|from|as|after|amid|with|and|following|valued|worth)\b)"
_PRICED = r"(?=\s+(?:for|in|at)\s+(?:\S+\s+){0,3}?(?:[$€£]\d|\d[\d.,]*\s+(?:million|billion|trillion)|deal\b))"
# A legal ending a headline may leave on the name. "Company" is not one: "target company" is anybody.
_ENDING = r"(?:,?\s+(?:inc|incorporated|corp|corporation|co|ltd|limited|plc|llc|holdings|group)\b\.?)?"
# First words too common to stand for one company ("General", "American").
_COMMON = frozenset(
    "advanced american applied bank china digital eastern energy first general global great home international "
    "national new northern pacific royal southern the united us western".split()
)
# A name is never cut short on one of these ("Bank of America", not "Bank of").
_JOINING = frozenset({"of", "and", "the", "de", "la"})
_LEGAL_ENDING = re.compile(r"[\s,]+(?:inc|corp|corporation|co|company|ltd|limited|plc|llc|lp|sa|nv|ag|se)\.?$", re.IGNORECASE)
MAX_NAME_WORDS = 5


def aliases(symbol: str, name: str | None = None) -> tuple[str, ...]:
    """What a headline may call a company: its ticker, and its name without the legal ending.

    An entry in capitals is a ticker and has to match exactly, so COST is not
    "cost". A one- or two-letter ticker is left out. The name is in lower case
    and matches in any case. How much of it a headline has to give is settled
    where it is matched (`_spelled`).
    """
    out = [symbol.upper()] if len(symbol) >= 3 and symbol.isalpha() else []
    words = re.findall(r"[a-z0-9][a-z0-9'&-]*", _LEGAL_ENDING.sub("", (name or "").split("/")[0].strip()).lower())
    if words[:1] == ["the"]:
        words = words[1:]
    if words:
        out.append(" ".join(words[:MAX_NAME_WORDS]))
    return tuple(out)


def _needed(words: list[str]) -> int | None:
    """How many words of a company's name a headline has to give. None if no part of it is safe to go by."""
    need = 2 if words[0] in _COMMON else 1
    while need <= len(words) and words[need - 1] in _JOINING:
        need += 1
    return None if need > len(words) or len(" ".join(words[:need])) < 3 else need


def _spelled(name: str) -> str | None:
    """A pattern for a company's name as a headline gives it, or None if no part of it is safe to go by.

    The first word has to be there and the rest may follow ("Pioneer", or
    "Pioneer Natural Resources"). A common first word needs the second as well
    ("General Motors", "Home Depot"), and a name is never cut short on a
    joining word ("Bank of America"). A name that is one common word names
    nobody.
    """
    words = name.split()
    need = _needed(words)
    if need is None:
        return None
    gap = r"(?:\s+&)?\s+"  # "Procter & Gamble" is on file as two words
    rest = ""
    for word in reversed(words[need:]):
        rest = rf"(?:{gap}{re.escape(word)}{rest})?"
    return gap.join(re.escape(word) for word in words[:need]) + rest


@lru_cache(maxsize=512)
def _bought(names: tuple[str, ...]) -> re.Pattern[str] | None:
    """Matches a headline in which one of `names` is the company being bought.

    The name has to come straight after the buying phrase, and the headline
    then has to end or turn to the terms. That is what tells "to Acquire Insmed
    for $41 Billion" from "to Acquire NVIDIA GPU Servers", which the same
    words would otherwise make a takeover of NVIDIA.
    """
    called = [n if n.isupper() else _spelled(n) for n in names]
    who = "|".join(re.escape(n) if n.isupper() else f"(?i:{n})" for n in called if n)
    if not who:
        return None
    bought = rf"\s+(?:all of\s+|the\s+)?(?-i:{who})\b{_ENDING}"
    return re.compile(rf"\b(?:(?:{_TAKING}){bought}{_THEN}|(?:{_PAYING}){bought}{_PRICED})", re.IGNORECASE)


# Words in a company's name that name nobody by themselves ("Palantir Technologies",
# "Costco Wholesale"), so a headline that has only that word does not name the company.
_PLAIN_WORDS = frozenset(
    "airlines america brands capital communications company corporation devices electric electronics enterprises "
    "express financial foods global group health healthcare holdings industries instruments international labs lines "
    "materials medical micro motor motors networks partners pharmaceuticals platforms resources sciences services "
    "software solutions stores systems technologies technology therapeutics wholesale".split()
)
# After the name these make someone else the subject: "Apple Supplier Foxconn Cuts Outlook".
_OF_ANOTHER = r"supplier|rival|competitor|partner|backer|investor|customer|peer|spin-?off"


@lru_cache(maxsize=512)
def _named(names: tuple[str, ...]) -> tuple[re.Pattern[str], re.Pattern[str]] | None:
    """Two patterns for a company: its name opening a headline, and its name anywhere in a text.

    None when no name is known, only a ticker. Headlines rarely give the
    ticker, so with nothing else to go by every headline would look like
    someone else's.

    Both are looser than the spelling a takeover needs (`_spelled`). They take
    the name with its dots and spaces moved ("J.P. Morgan" and "JPMorganChase"
    for "jpmorgan chase") and a later word of the name alone ("Lilly",
    "Chase"), unless that word is a plain one. Looser is the safe side here:
    a headline wrongly taken to name the company is typed as it always was.
    """
    ways = []
    for name in names:
        words = name.split()
        need = None if name.isupper() else _needed(words)
        if need is None:
            continue
        gap = r"[\s&.,]*"
        loosely = [r"[.\s]*".join(map(re.escape, word)) if len(word) >= 6 else re.escape(word) for word in words]
        rest = ""
        for word in reversed(loosely[need:]):
            rest = rf"(?:{gap}{word}{rest})?"
        ways.append(gap.join(loosely[:need]) + rest)
        ways += [re.escape(w) for w in words[need:] if len(w) >= 5 and w not in _COMMON and w not in _PLAIN_WORDS]
    if not ways:
        return None
    tickers = [re.escape(n) for n in names if n.isupper()]
    who = "|".join([f"(?i:{way})" for way in ways] + tickers)
    opens = re.compile(rf"(?:(?i:the)\s+)?(?:{who})(?![A-Za-z0-9-])(?!\s+(?i:{_OF_ANOTHER})\b)")
    anywhere = re.compile(rf"(?<![A-Za-z0-9])(?:{who})(?![A-Za-z0-9])")
    return opens, anywhere


# What puts the company that opens a headline on the receiving end of a deal. Without one of
# these it may be the bidder: "GameStop Makes Unsolicited $56 Billion Takeover Bid for eBay".
_RECEIVING = re.compile(
    r"\b(?:to be (?:acquired|bought|taken private|sold)|agrees? to be|(?:acquired|bought out|taken private) by"
    r"|receiv\w+ (?:(?!approv)\S+ ){0,4}?(?:offer|bid|proposal|approach|interest)\b"
    r"|reject\w+ (?:(?!approv)\S+ ){0,6}?(?:offer|bid|proposal|approach)\b"
    r"|(?:offer|bid|proposal|approach|interest) from\b"
    r"|go(?:es|ing)? private|to go private|take[- ]private|buyout by\b"
    r"|strategic (?:alternatives|review)|explor\w+ (?:a |potential )?sale|sale of the company"
    r"|to (?:merge|combine) with|merger (?:agreement|with)\b|agrees? to merge)",
    re.IGNORECASE,
)
# What may stand between a company's name and its verb: "JPMorgan Chase & Co. Adjusts ...".
_UNIT = r"(?:\s*&\s*co\b\.?|,?\s+(?:inc|corp|co|ltd|plc|llc|group|holdings|securities|research)\b\.?)*"
# The verbs an analyst's note opens with. Straight after the company's own name they make
# it the writer of the note, and the news is about the company it rates.
_WRITES = re.compile(
    _UNIT + r"\s+(?:maintains?|reiterates?|reaffirms?|raises?|lowers?|cuts?|adjusts?|initiates?|resumes?|starts?"
    r"|upgrades?|downgrades?|updates?|assigns?|sets?|boosts?|trims?|keeps?)\b",
    re.IGNORECASE,
)
# A holding in a company. When the company that opens the headline holds it in someone
# else, the trade is in that other company's shares.
_HOLDS = re.compile(r"\b(?:stake|holdings?|shareholding|voting rights)\s+in\s+(.*)", re.IGNORECASE)
_ITSELF = re.compile(r"(?:the company|itself|its own)\b", re.IGNORECASE)


def headline_type(text: str, names: tuple[str, ...] = (), summary: str = "") -> str:
    """A headline's event type. `names` are what the company may be called (see `aliases`).

    The feed tags the buyer, its bankers and the company being bought with the
    same headline, and the keyword rules cannot tell them apart. With the
    company's own name to go on, "Sanofi to Acquire Insmed" is a takeover for
    Insmed and an acquisition for everyone else, which matters because a
    buyout is the case where the scan's rating stops meaning anything.

    This check comes ahead of the table on purpose. A buyout headline often
    ends "... Shares Jump", which the table would read as someone trading
    shares.

    With a name to go by, a type counts for the company only when the headline
    puts the company where that event happens:

    - A headline that names the company nowhere, and whose `summary` does not
      either, is another company's news. With no summary on hand nothing is
      concluded, because a headline often names only a product.
    - The company holding a stake in someone else, or writing an analyst's
      note about someone else, is another company's news too.
    - A type in `IN_FRONT` needs the company's name to open the headline.
    - A takeover by the table's keywords needs the company to open the
      headline and be on the receiving end (`_RECEIVING`).

    Where one of the last two does not hold, the rest of the table decides.
    """
    bought = _bought(names)
    if bought is not None and bought.search(text):
        return "takeover"
    known = _named(names)
    lead = None
    if known is not None:
        opens, anywhere = known
        if anywhere.search(text):
            lead = opens.match(text)
        elif summary and not anywhere.search(summary):
            return ELSEWHERE
        if lead:
            held = _HOLDS.search(text, lead.end())
            if held and not anywhere.search(held.group(1)) and not _ITSELF.match(held.group(1)):
                return ELSEWHERE
    for kind, pattern in HEADLINE_RULES:
        if not pattern.search(text):
            continue
        if known is not None:
            if kind in IN_FRONT and lead is None:
                continue
            if kind == "takeover" and not (lead and _RECEIVING.search(text, lead.end())):
                continue
            if kind == "analyst" and lead and _WRITES.match(text, lead.end()):
                return ELSEWHERE
        return kind
    return "other"


def classify(item: NewsItem, names: tuple[str, ...] = ()) -> NewsItem:
    """The item with its event type set. A filing also gets a title that spells out its items."""
    if item.kind != "filing":
        return replace(item, event=headline_type(item.title, names, item.summary))
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


def _placed(items: list[NewsItem], days: list[date], earnings_days: set[date], kind: str) -> dict[int, tuple[int, NewsItem]]:
    """One type's events as sessions: the session that reflects each, with the session before it and the earliest item."""
    sessions: dict[int, tuple[int, NewsItem]] = {}
    for item in sorted((i for i in items if i.event == kind), key=lambda i: i.when):
        found = window(*_felt(item.when), days)
        if found is None or days[found[1]] in earnings_days:
            continue
        sessions.setdefault(found[1], (found[0], item))
    return sessions


def _stretch(sessions: dict[int, tuple[int, NewsItem]], days: list[date]) -> int:
    """The session a type's usual day is measured from: its first event, or `BASELINE_SESSIONS` back if that is earlier."""
    return max(1, min(min(sessions), len(days) - BASELINE_SESSIONS))


def _usual(start: int, days: list[date], closes: list[float], earnings_days: set[date]) -> list[float]:
    """The move on every session from `start` on, earnings days left out."""
    return [closes[i] / closes[i - 1] - 1.0 for i in range(start, len(days)) if days[i] not in earnings_days]


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
    length in the stretch. A span with an earnings reaction in it is left out
    on both sides, like the day itself: an event three sessions before a
    report would otherwise be credited with the report's jump.

    "Any session" includes the event days themselves. Leaving them out would
    make a cleaner contrast for a rare event, but for a company in the news
    most days (seen live: 114 of 145 sessions) it would leave a few odd days to
    stand for the ordinary one.
    """
    days = [b.day for b in bars]
    closes = [b.close for b in bars]

    def followed_by(i: int) -> float | None:
        """The move over the sessions after session `i`, if they are on record and hold no report."""
        later = i + FOLLOW_SESSIONS
        if later >= len(closes) or any(day in earnings_days for day in days[i + 1 : later + 1]):
            return None
        return closes[later] / closes[i] - 1.0

    out: dict[str, EventStudy] = {}
    for kind in dict.fromkeys(kinds):
        if kind in NOT_STUDIED:
            continue
        sessions = _placed(items, days, earnings_days, kind)
        if not sessions:
            continue

        moves = []
        for after, (before, item) in sorted(sessions.items(), reverse=True):
            moves.append(
                EventMove(
                    day=days[after],
                    title=item.title,
                    move_pct=closes[after] / closes[before] - 1.0,
                    after_pct=followed_by(after),
                )
            )
        start = _stretch(sessions, days)
        ordinary = _mean_abs(_usual(start, days, closes, earnings_days))
        followed = [m.after_pct for m in moves if m.after_pct is not None]
        spans = [span for span in map(followed_by, range(start, len(days))) if span is not None]
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
            ordinary_abs_after=_mean_abs(spans) if followed else None,
            moves=moves[:MAX_MOVES],
        )
    return out


def coverage(items: Iterable[NewsItem], today: date) -> str:
    """Which stocks a stock is compared with: "heavy" if the feed carries news of its own most days, else "light".

    The cut matters more than anything else in a pooled study. Measured on 73
    stocks: for a heavily covered one no type of headline was followed by a
    bigger day than usual, while for a lightly covered one an offering or a
    reverse split was. Pooled together the two cancel out. `items` are
    classified, so another company's news is not counted towards coverage.
    """
    cutoff = today - timedelta(days=COVERAGE_DAYS)
    own = sum(1 for i in items if i.kind == "headline" and i.event != ELSEWHERE and _felt(i.when)[0] >= cutoff)
    return "heavy" if own >= HEAVY_HEADLINES else "light"


def tally(items: list[NewsItem], bars: list[Bar], earnings_days: set[date]) -> dict[str, list[int]]:
    """What one stock adds to a pool, by event type: `TALLY` counts for each type it has events of.

    A day is big when it moved more than `BIG_MOVE` times the stock's usual
    day, which is measured as in `study`: over the stretch the type's events
    span, earnings days left out. Counting against each stock's own usual day
    is what lets a volatile stock and a quiet one be added together.
    """
    days = [b.day for b in bars]
    closes = [b.close for b in bars]
    out: dict[str, list[int]] = {}
    for kind in dict.fromkeys(i.event for i in items):
        if kind in NOT_STUDIED:
            continue
        sessions = _placed(items, days, earnings_days, kind)
        if not sessions:
            continue
        usual = _usual(_stretch(sessions, days), days, closes, earnings_days)
        size = _mean_abs(usual)
        if not size:
            continue  # a stock that never moved has no usual day to measure against
        moves = [closes[after] / closes[before] - 1.0 for after, (before, _) in sessions.items()]
        out[kind] = [
            len(moves),
            sum(1 for m in moves if abs(m) > BIG_MOVE * size),
            sum(1 for m in moves if m > 0),
            sum(1 for m in moves if m < 0),
            len(usual),
            sum(1 for m in usual if abs(m) > BIG_MOVE * size),
        ]
    return out


def pooled(totals: dict[str, dict], group: str, kinds: Iterable[str]) -> dict[str, PooledStudy]:
    """Pooled studies for one group of stocks, for each type in `kinds` that the pool has any event of.

    `totals` is what `EventPool.totals` returns: for each group, how many
    stocks are on record and, by type, the summed `TALLY` counts and how many
    stocks they came from.
    """
    found = totals.get(group) or {}
    out: dict[str, PooledStudy] = {}
    for kind in dict.fromkeys(kinds):
        row = (found.get("kinds") or {}).get(kind)
        if kind in NOT_STUDIED or not row or not row["counts"][0]:
            continue
        events, big, up, down, sessions, big_sessions = row["counts"]
        out[kind] = PooledStudy(
            event=kind,
            label=EVENT_TYPES[kind],
            group=group,
            group_stocks=found.get("stocks", 0),
            stocks=row["stocks"],
            events=events,
            big=big,
            up=up,
            down=down,
            sessions=sessions,
            big_sessions=big_sessions,
        )
    return out


def shortlist(items: Iterable[NewsItem], limit: int) -> list[NewsItem]:
    """Up to `limit` of `items`, newest first, for a surface with room for only a few.

    The newest item of each type that carries a caution is taken first, then
    the newest of each other flagged type, then whatever is newest. So a
    caution is never shown without the headline behind it, however many newer
    items there are and whatever their types.
    """
    newest = sorted(items, key=lambda i: i.when, reverse=True)
    first = [*CAUTIONS, *(kind for kind in EVENT_TYPES if kind in FLAGGED and kind not in CAUTIONS)]
    lead = [item for item in (next((i for i in newest if i.event == kind), None) for kind in first) if item is not None]
    taken = {id(item) for item in lead[:limit]}
    rest = [item for item in newest if id(item) not in taken]
    return sorted(lead[:limit] + rest[: max(0, limit - len(taken))], key=lambda i: i.when, reverse=True)


class NewsContext(NamedTuple):
    """What `context` works out for one stock."""

    shown: list[NewsItem]  # the latest items of its own, newest first
    counts: dict[str, int]  # recent items by event type
    studies: dict[str, EventStudy]  # what followed its past events, by type
    elsewhere: list[NewsItem]  # recent items that are another company's news
    coverage: str  # "light" or "heavy": which stocks it is pooled with
    tally: dict[str, list[int]]  # what it adds to that pool, by type


def context(
    items: list[NewsItem], bars: list[Bar], earnings_days: set[date], today: date, names: tuple[str, ...] = ()
) -> NewsContext:
    """One stock's news for a scan: the latest items, a count of recent ones by type, the studies,
    the recent items that are another company's news, and its place in a pool.

    `items` are everything on record for the stock, classified or not, and
    `names` are what a headline may call the company. A study is built for
    each type seen in the last `RECENT_DAYS` days, from the whole record.

    The list holds the newest `MAX_RECENT` items, plus the newest item of each
    flagged type that a busy week pushed out of them. A chip that says
    "takeover" is no use if the headline behind it is not there to read.

    Another company's news takes none of those places. It is counted, so the
    page can say how much of it there is, and the newest `MAX_ELSEWHERE` are
    handed back apart, to be looked at by anyone who doubts the rule.

    The tally is over the whole record and every type in it, not only the
    recent ones: it is this stock's share of a pool that other stocks draw on.
    """
    classified = sorted((classify(i, names) for i in items), key=lambda i: i.when, reverse=True)
    cutoff = today - timedelta(days=RECENT_DAYS)
    recent = [i for i in classified if _felt(i.when)[0] >= cutoff]
    counts = Counter(i.event for i in recent)
    order = [kind for kind in EVENT_TYPES if kind in counts]
    own = [i for i in recent if i.event != ELSEWHERE]
    shown = own[:MAX_RECENT]
    listed = {i.event for i in shown}
    shown += [next(i for i in own if i.event == kind) for kind in order if kind in FLAGGED and kind not in listed]
    shown.sort(key=lambda i: i.when, reverse=True)
    elsewhere = [i for i in recent if i.event == ELSEWHERE][:MAX_ELSEWHERE]
    return NewsContext(
        shown=shown,
        counts={kind: counts[kind] for kind in order},
        studies=study(classified, bars, earnings_days, order),
        elsewhere=elsewhere,
        coverage=coverage(classified, today),
        tally=tally(classified, bars, earnings_days),
    )
