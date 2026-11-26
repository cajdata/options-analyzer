"""News layer: event types, what followed past events, and the sources behind them."""

import json
from dataclasses import replace
from datetime import date, datetime, time as clock, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.analysis import events
from app.cache import DiskCache, NullCache
from app.models import Bar, NewsItem
from app.providers.alpaca import MAX_NEWS_PAGES, SUMMARY_CHARS, AlpacaData, AlpacaError
from app.providers.earnings import EdgarEarnings, Unsupported
from app.providers.news import NewsService

FIXTURES = Path(__file__).parent / "fixtures"
NEWS = json.loads((FIXTURES / "alpaca_news.json").read_text(encoding="utf-8"))
ET = ZoneInfo("America/New_York")
TODAY = date(2026, 11, 24)


# -- Rules ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "headline, kind",
    [
        ("Acme Reports Third Quarter 2026 Financial Results", "earnings"),
        ("Acme Reports Loss on Plant Closure but Lifts 2026 Financial Forecast", "earnings"),
        ("Acme Raises Full-Year 2026 Revenue Guidance", "guidance"),
        ("Acme Warns of Margin Pressures in the December Quarter", "guidance"),
        ("Quillfeather Research Raises Acme Price Target to $229", "analyst"),
        ("Marrowgate Capital Initiates Coverage of Acme with Buy Rating", "analyst"),
        ("Acme Agrees to Be Acquired by Tidewell Systems for $4 Billion", "takeover"),
        ("Acme Receives Unsolicited Acquisition Proposal", "takeover"),
        ("Acme Acquires Tidewell Systems for Up to $2.3 Billion", "acquisition"),
        ("Acme Announces 4-for-1 Stock Split", "split"),
        ("Acme Declares Special Cash Dividend", "split"),
        ("Acme Announces Pricing of Upsized $2.6 Billion Offering of Convertible Senior Notes", "financing"),
        ("Acme Announces 10% Dividend Increase", "payout"),
        ("Acme CFO Sells Shares Worth Over $1 Million", "insider"),
        ("Longview Partners Increases Stake in Acme to 6.8%", "insider"),
        ("Acme Files for Chapter 11 Bankruptcy Protection", "distress"),
        ("Acme Reports Cyberattack on Third-Party Vendors", "incident"),
        ("Acme Announces Positive Phase 3 Results for Its Lung Cancer Drug", "drug"),
        ("FDA Approves Acme's Combination Treatment", "drug"),
        ("Acme Recalls Over 565,000 Vehicles Due to Fire Risk", "legal"),
        ("Acme Faces Civil Trial Over Product Claims", "legal"),
        ("Acme Conducts Targeted Layoffs in Its Fitness Division", "restructuring"),
        ("Acme Shareholders Approve Charter Amendment", "governance"),
        ("Acme Appoints Dana Whitlow as Chief Financial Officer", "management"),
        ("Acme Software Chief Jon Marlowe Departs", "management"),
        ("Acme and Tidewell Systems Announce Multi-Year Partnership", "agreement"),
        ("Acme Launches Next-Generation Platform", "product"),
        ("Acme to Present at Lakeshore Technology Conference", "calendar"),
        ("Acme Publishes Annual Sustainability Report", "other"),
    ],
)
def test_headlines_are_sorted_into_event_types_by_keyword(headline, kind):
    assert events.headline_type(headline) == kind


@pytest.mark.parametrize(
    "headline, kind",
    [
        # Raising a price target is an analyst's note, not the company's guidance.
        ("Quillfeather Research Raises Acme Price Target Ahead of Q1 Earnings", "analyst"),
        # A notice of a date is not the results.
        ("Acme Announces Webcast for Third Quarter 2026 Earnings Conference Call", "calendar"),
        ("Acme Schedules Third Quarter 2026 Financial Results Release", "calendar"),
        # "to Hold" is a rating here, not a meeting being held.
        ("Marrowgate Capital Downgrades Acme to Hold From Buy", "analyst"),
        # Weight loss is not a loss.
        ("Acme Reports Its New Pill Showed Greater Weight Loss Than a Rival in Data Analysis", "other"),
        # A study the company publishes about its market is not its own outlook.
        ("Acme Releases 2026 Commercial Market Outlook", "product"),
        # The lender is not the one raising money.
        ("Acme Bank Leads $660 Million Credit Facility for Tidewell Systems", "agreement"),
        ("Acme Ventures Participates in Tidewell's Series A Financing Round", "agreement"),
        # A bond leaving an exchange is routine; a stock leaving one is not.
        ("Acme Delists 1.625% Notes Due 2026 from NYSE", "other"),
        ("Acme Receives Nasdaq Delisting Notice", "distress"),
        # A clinical trial is not a court case.
        ("Acme Initiates Phase 3 Trial for Its Flu Shot", "drug"),
        # An executive selling shares is a trade, not a change of management.
        ("Acme Director Files Form 144 to Propose Sale of 6.42 Million Shares", "insider"),
    ],
)
def test_the_order_of_the_rules_settles_a_headline_that_fits_two(headline, kind):
    assert events.headline_type(headline) == kind


@pytest.mark.parametrize(
    "headline, kind",
    [
        # The default search engine on a phone is not a default on a debt.
        ("Tidewell Pays Acme to Remain Default Search Engine on Its Phones", "other"),
        ("Acme Defaults on $40 Million Bond Payment", "distress"),
        ("Tidewell Resolves Alleged Default on Acme Offtake Agreement", "distress"),
        ("Acme Receives Notice of Default from Its Lenders", "distress"),
        # A company prices a product as well as an offering.
        ("Acme Prices Its New Headset at $3,499", "other"),
        ("Acme Prices $500,000 in Callable Fixed-to-Floating Notes", "financing"),
        ("Acme Prices Upsized Offering", "financing"),
        ("Acme Prices Its Initial Public Offering at $21 a Share", "financing"),
        # Positive data for a chip is not a drug trial.
        ("Acme Reports Positive Benchmark Data for Its New Processor", "other"),
        ("Acme Announces Positive 12-Month Data from Open-Label Extension Study of Its Inhaler", "drug"),
        ("Acme Reports Positive Seven-Year Follow-up Data in Lung Cancer Trial", "drug"),
        ("Acme Announces Positive Phase 1b Results for AC-102", "drug"),
        # A test programme has phases too.
        ("Acme Receives FAA Approval for Phase 4B Certification Testing", "legal"),
    ],
)
def test_a_flagged_type_needs_more_than_one_loose_word(headline, kind):
    assert events.headline_type(headline) == kind


@pytest.mark.parametrize(
    "symbol, name, expected",
    [
        ("INSM", "INSMED Inc", ("INSM", "insmed")),
        ("JPM", "JPMORGAN CHASE & CO", ("JPM", "jpmorgan chase")),
        ("COST", "COSTCO WHOLESALE CORP /NEW", ("COST", "costco wholesale")),
        ("CRM", "Salesforce, Inc.", ("CRM", "salesforce")),
        ("F", "Ford Motor Co", ("ford motor",)),  # a one-letter ticker would match far too much
        ("BAC", "BANK OF AMERICA CORP /DE/", ("BAC", "bank of america")),
        ("CI", "The Cigna Group", ("cigna group",)),
        ("BRK.B", "BERKSHIRE HATHAWAY INC", ("berkshire hathaway",)),
        ("ZZZZ", None, ("ZZZZ",)),
        ("T", "", ()),
    ],
)
def test_a_company_goes_by_its_ticker_and_its_name(symbol, name, expected):
    assert events.aliases(symbol, name) == expected


@pytest.mark.parametrize(
    "name, said, not_said",
    [
        # The first word is enough, and the rest of the name may follow it.
        ("pioneer natural resources", ["Pioneer", "Pioneer Natural", "Pioneer Natural Resources"], ["Natural Resources"]),
        # A common first word needs the second as well.
        ("general motors", ["General Motors"], ["General"]),
        ("home depot", ["Home Depot"], ["Home"]),
        # A name is never cut short on a joining word.
        ("bank of america", ["Bank of America"], ["Bank of", "Bank"]),
        ("procter gamble", ["Procter & Gamble", "Procter"], ["Gamble"]),
    ],
)
def test_how_much_of_a_name_a_headline_has_to_give(name, said, not_said):
    import re

    pattern = re.compile(events._spelled(name), re.IGNORECASE)
    assert all(pattern.fullmatch(text) for text in said)
    assert not any(pattern.fullmatch(text) for text in not_said)
    # One common word names nobody, and nor does a word of one or two letters.
    assert events._spelled("united") is None and events._spelled("x") is None


def test_the_company_being_bought_is_told_from_the_buyer_by_name():
    insmed, sanofi = ("INSM", "insmed"), ("SNY", "sanofi")
    deal = "Sanofi to Acquire Insmed for $41 Billion"
    # The feed tags both companies with this headline. Only one of them is being taken over.
    assert events.headline_type(deal, insmed) == "takeover"
    assert events.headline_type(deal, sanofi) == "acquisition"
    assert events.headline_type(deal) == "acquisition"  # with no name to go on, the keyword rules decide
    assert events.headline_type("Sanofi Reportedly Considering Acquisition of Insmed", insmed) == "takeover"
    assert events.headline_type("Ademi LLP Investigates Proposed Acquisition of Insmed", insmed) == "takeover"
    assert events.headline_type("Tidewell Systems Makes $9 Billion Offer for INSM", insmed) == "takeover"
    # Insmed as the buyer, and a deal that only mentions it, are not takeovers of Insmed.
    assert events.headline_type("Insmed to Acquire Tidewell Systems", insmed) == "acquisition"
    assert events.headline_type("Berenberg Reaffirms Sanofi Buy Rating Amid Ongoing Acquisition Rumors", insmed) == "analyst"

    # A ticker that is also a word has to appear as a ticker.
    costco = ("COST", "costco")
    assert events.headline_type("Tidewell Systems Wins Deal for Cost Monitoring Software", costco) == "agreement"
    assert events.headline_type("Tidewell Systems to Buy COST in $400 Billion Deal", costco) == "takeover"
    assert events.classify(headline(1, NOON, deal), insmed).event == "takeover"


@pytest.mark.parametrize(
    "symbol, name, text",
    [
        # Seen live: all seven were typed as a takeover of the company named.
        ("NVDA", "NVIDIA CORP", "Quantum Cyber Acquires NVIDIA-Powered AI Compute Cluster"),
        ("NVDA", "NVIDIA CORP", "GMO Internet Board Approves 10 Billion Yen Deal to Acquire NVIDIA GPU Servers"),
        ("NVDA", "NVIDIA CORP", "Digi Power X Commits $35 Million to Acquire NVIDIA Vera Rubin Systems"),
        ("PFE", "PFIZER INC", "Novartis India Acquires Pfizer Brands for $132 Million"),
        ("BA", "BOEING CO", "China Advances on Commitment to Purchase Boeing Jets"),
        ("BA", "BOEING CO", "Archer Aviation to Acquire Boeing Subsidiaries Wisk Aero, SkyGrid, and Insitu"),
        # Its shares, its products or one of its units is not the company.
        ("NVDA", "NVIDIA CORP", "Tidewell to Buy Nvidia Chips in $10 Billion Deal"),
        ("BA", "BOEING CO", "Tidewell Air Agrees to Buy Boeing 737 MAX Jets"),
        ("AAPL", "Apple Inc.", "Longview Partners Buys Apple Shares in Third Quarter"),
        ("TSLA", "Tesla, Inc.", "Longview Partners Buys Tesla Stock After Dip"),
        ("MSFT", "MICROSOFT CORP", "Director Reports Purchase of Microsoft Shares"),
        ("INSM", "INSMED Inc", "Sanofi to Acquire Insmed's Lung Division"),
        # Advice to buy the stock is not an offer for the company.
        ("NVDA", "NVIDIA CORP", "Why Investors Should Look to Buy NVDA Ahead of Earnings"),
        ("AMZN", "AMAZON COM INC", "Marrowgate Capital Reiterates Outperform, Tells Clients to Buy Amazon"),
        ("AMZN", "AMAZON COM INC", "Three Reasons to Buy Amazon for the Long Term"),
        # Another company whose name starts the same way.
        ("BAC", "BANK OF AMERICA CORP /DE/", "Tidewell Capital to Acquire Bank of Hawaii"),
        ("HD", "HOME DEPOT, INC.", "Tidewell to Acquire Home Services Startup"),
        ("TGT", "TARGET CORP", "Tidewell to Acquire Target Company in All-Cash Deal"),
        ("TXN", "TEXAS INSTRUMENTS INC", "Tidewell to Acquire Texas Capital Bank for $2 Billion"),
    ],
)
def test_buying_something_of_a_company_is_not_buying_the_company(symbol, name, text):
    assert events.headline_type(text, events.aliases(symbol, name)) != "takeover"


@pytest.mark.parametrize(
    "symbol, name, text",
    [
        ("HES", "HESS CORP", "Tidewell to Buy Hess for $53 Billion"),
        ("HES", "HESS CORP", "Tidewell to Buy Hess in All-Stock Deal"),
        ("HES", "HESS CORP", "Tidewell Agrees to Buy Hess"),
        ("HES", "HESS CORP", "Tidewell Completes Acquisition of Hess Corp."),
        ("HES", "HESS CORP", "Tidewell to Acquire Hess, Expanding in Guyana"),
        ("HES", "HESS CORP", "Tidewell Raises Bid for Hess to $171 a Share"),
        ("PXD", "PIONEER NATURAL RESOURCES CO", "Tidewell to Acquire Pioneer Natural Resources for $60 Billion"),
        ("PXD", "PIONEER NATURAL RESOURCES CO", "Tidewell to Acquire Pioneer in $60 Billion Deal"),
        ("BAC", "BANK OF AMERICA CORP /DE/", "Tidewell Capital Makes Offer for Bank of America"),
        ("HD", "HOME DEPOT, INC.", "Tidewell to Take Private Home Depot Inc. for $400 Billion"),
        ("LLY", "ELI LILLY & Co", "Tidewell to Acquire Eli Lilly (LLY) at $900 a Share"),
    ],
)
def test_a_takeover_is_still_caught_however_the_headline_goes_on(symbol, name, text):
    assert events.headline_type(text, events.aliases(symbol, name)) == "takeover"


COMPANIES = {
    "GME": "GameStop Corp.",
    "EBAY": "eBay Inc.",
    "COST": "COSTCO WHOLESALE CORP /NEW",
    "NVDA": "NVIDIA CORP",
    "TSLA": "Tesla, Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "LLY": "ELI LILLY & Co",
    "JPM": "JPMORGAN CHASE & CO",
    "AAPL": "Apple Inc.",
    "PFE": "PFIZER INC",
    "ACME": "Acme",
}


# Brands a registered name does not give, as data/aliases.txt lists them.
LISTED = {"C": ("Citi", "Citibank"), "SPCX": ("SpaceX",), "NU": ("Nubank",), "META": ("Facebook", "Instagram", "WhatsApp")}


def typed(symbol: str, text: str, summary: str = "") -> str:
    also = events.also_called(LISTED.get(symbol, ()))
    return events.headline_type(text, events.aliases(symbol, COMPANIES[symbol]), summary, also)


@pytest.mark.parametrize(
    "symbol, text, kind",
    [
        # Seen live: GameStop bid for eBay, and every one of these marked GameStop as the company being bought.
        ("GME", "GameStop Makes Unsolicited $56 Billion Takeover Bid for eBay", "other"),
        ("GME", "GameStop Formally Submits Unsolicited $55.5 Billion Acquisition Proposal for eBay", "acquisition"),
        ("GME", "GameStop Shareholders Approve Measures Supporting eBay Acquisition Bid", "governance"),
        ("GME", "eBay Rejects GameStop's $55.5 Billion Acquisition Proposal", "acquisition"),
        ("GME", "Michael Burry Exits GameStop Position Following eBay Acquisition Bid", "acquisition"),
        # The same headlines are a takeover for the company on the receiving end.
        ("EBAY", "GameStop Makes Unsolicited $56 Billion Takeover Bid for eBay", "takeover"),
        ("EBAY", "eBay Rejects GameStop's $55.5 Billion Acquisition Proposal", "takeover"),
        ("EBAY", "eBay Board of Directors Rejects GameStop's Unsolicited Acquisition Proposal", "takeover"),
        ("ACME", "Acme Agrees to Be Acquired by Tidewell Systems for $4 Billion", "takeover"),
        ("ACME", "Acme Receives Unsolicited Acquisition Proposal", "takeover"),
        ("ACME", "Acme to Go Private in $2 Billion Deal", "takeover"),
        ("ACME", "Acme Reviews Strategic Alternatives", "takeover"),
        # Receiving a vote is not receiving an offer, and a product "takeover" in a shop is not a deal.
        ("ACME", "Acme Receives Shareholder Approval for Tidewell Acquisition Bid", "acquisition"),
        ("COST", "Fly By Jing Launches Chili Crisp Takeover at Costco", "product"),
    ],
)
def test_a_takeover_needs_the_company_on_the_receiving_end(symbol, text, kind):
    assert typed(symbol, text) == kind


@pytest.mark.parametrize(
    "symbol, text, kind",
    [
        # Seen live: someone else raising money, with the stock named further on.
        ("NVDA", "IREN Secures $3.6 Billion Financing for NVIDIA GPU Acquisition", "acquisition"),
        ("NVDA", "NVIDIA-Backed Iambic Therapeutics Files for US IPO", "other"),
        ("NVDA", "NVIDIA Customer AM Intelligence Seeks $3 Billion Financing for Processor Order", "agreement"),
        ("JPM", "Aviation Capital Group Amends Credit Facility with JPMorgan Chase", "other"),
        ("NVDA", "NVIDIA Raises $25 Billion in Corporate Bond Offering", "financing"),
        ("JPM", "JPMorgan Chase Prices $3.8 Million in Callable Fixed-Rate Notes", "financing"),
        # An analyst's forecast for the company is not the company's guidance.
        ("TSLA", "Morgan Stanley Raises Tesla Q2 Delivery Estimate", "other"),
        ("AAPL", "Apple Supplier Foxconn Cuts Full-Year Revenue Outlook", "other"),
        ("PLTR", "Palantir Raises Full-Year 2026 Revenue Outlook Following Strong Q1 Performance", "guidance"),
        ("NVDA", "NVIDIA Warns of Potential Second-Half Margin Compression", "guidance"),
        # A later word of the name opens the headline as well as the whole of it.
        ("LLY", "Lilly Raises Full-Year Revenue Guidance", "guidance"),
        # Talk of a special dividend is not one, and another company's default is not this one's.
        ("COST", "Mizuho Analyst Suggests Costco Reinvest Capital Instead of Special Dividend", "payout"),
        ("COST", "Costco Declares Special Cash Dividend of $15 Per Share", "split"),
        ("TSLA", "Syrah Resources Resolves Alleged Default on Tesla Offtake Agreement", "agreement"),
        # Drug news is the exception: the regulator opens the headline, and the news is the company's.
        ("LLY", "FDA Approves Eli Lilly's Mounjaro to Reduce Cardiovascular Risk in Type 2 Diabetes", "drug"),
    ],
)
def test_some_flagged_types_need_the_company_to_open_the_headline(symbol, text, kind):
    assert typed(symbol, text) == kind
    assert events.IN_FRONT == {"guidance", "split", "financing", "distress"} and events.IN_FRONT < events.FLAGGED


@pytest.mark.parametrize(
    "symbol, text, kind",
    [
        # Seen live: a third of JPMorgan's headlines. It wrote the note, or it holds the shares.
        ("JPM", "JPMorgan Chase Maintains Overweight Rating on T-Mobile US", "another"),
        ("JPM", "JPMorgan Chase & Co. Adjusts Price Target for Truist Financial", "another"),
        ("JPM", "J.P. Morgan Updates Rating and Price Target for Costco", "another"),
        ("COST", "J.P. Morgan Updates Rating and Price Target for Costco", "analyst"),
        ("JPM", "Barclays Adjusts Price Target on JPMorgan Chase to $425 From $420", "analyst"),
        ("JPM", "JPMorgan Chase Increases Stake in Future PLC to 6.37%", "another"),
        ("JPM", "JPMorgan Chase Updates Major Holdings in Siltronic AG to 5.21%", "another"),
        # Someone holding this company's shares, and the company's own people, are still its insider news.
        ("ACME", "Longview Partners Increases Stake in Acme to 6.8%", "insider"),
        ("ACME", "Acme Chairman Raises Stake in Acme", "insider"),
        # A banker to someone else's listing is doing business, not raising money.
        ("JPM", "JPMorgan Chase Selected to Lead General Atlantic IPO", "agreement"),
        ("NVDA", "NVIDIA Pledges $10 Billion to OpenAI Funding Round", "agreement"),
        # Words that only look like the shapes above.
        ("AAPL", "Apple Updates App Store Age-Rating Requirements for Dating and AI Companion Apps", "other"),
        ("TSLA", "Tesla Maintains Top Position in South Korea's Imported Car Market", "other"),
    ],
)
def test_a_company_acting_on_another_is_not_having_the_event(symbol, text, kind):
    assert typed(symbol, text) == kind
    assert events.ELSEWHERE == "another" and events.ELSEWHERE in events.NOT_STUDIED and events.ELSEWHERE not in events.FLAGGED


def test_a_headline_that_never_names_the_company_is_another_companys_news():
    # Seen live: 52 of Costco's 314 headlines were Costain's, a company whose ticker is COST in London.
    costain = "Costain Group PLC Announces Total Voting Rights"
    assert typed("COST", costain, "Costain Group PLC reported its total voting rights and capital.") == "another"
    # The summary can name the company where the headline has only a product.
    mounjaro = "Health Canada Approves Mounjaro for Cardiovascular Risk Reduction"
    assert typed("LLY", mounjaro, "Health Canada approved Eli Lilly's Mounjaro for a new use.") == "other"
    assert typed("LLY", mounjaro, "Health Canada approved Mounjaro for a new use.") == "another"
    assert typed("AAPL", "Analysts Project Potential Price Increase for iPhone 18 Series", "Analysts expect Apple to raise prices.") == "other"

    # With no summary on hand nothing is concluded: a headline alone often has only the product.
    assert typed("COST", costain) == "other"
    # With only a ticker to go by, nothing is either. Headlines rarely give the ticker.
    assert events.headline_type(costain, ("COST",), "Costain Group PLC reported its total voting rights.") == "other"
    assert events._named(("COST",)) is None and events._named(()) is None

    # A name is found with its dots and spaces moved, and by a later word that is not a plain one.
    said = "The bank's arm announced it."
    assert typed("JPM", "J.P. Morgan Payments Partners with Mercado Libre for Global Selling Program", said) == "agreement"
    assert typed("JPM", "JPMorganChase Announces $10 Million Loan to Support Atlanta Small Businesses", said) == "other"
    assert typed("JPM", "Chase Travel Announces Expansion of 'The Edit' Collection", said) == "other"
    assert typed("PLTR", "Plus Therapeutics Partners with Ephemeral Technologies for an Oncology Platform", "Two companies partnered.") == "another"


def test_with_no_name_to_go_by_the_keyword_table_decides_as_before():
    # No company named: a takeover word is a takeover, and an offering is an offering, whoever it is.
    assert events.headline_type("GameStop Makes Unsolicited $56 Billion Takeover Bid for eBay") == "takeover"
    assert events.headline_type("IREN Secures $3.6 Billion Financing for NVIDIA GPU Acquisition") == "financing"
    assert events.headline_type("JPMorgan Chase Increases Stake in Future PLC to 6.37%") == "insider"


COMPANIES.update(
    {
        "SHOP": "SHOPIFY INC.",
        "UBER": "Uber Technologies, Inc",
        "SNDK": "Sandisk Corp",
        "GS": "GOLDMAN SACHS GROUP INC",
        "DIS": "Walt Disney Co",
        "AMD": "ADVANCED MICRO DEVICES INC",
        "AVGO": "Broadcom Inc.",
        "CRWV": "CoreWeave, Inc.",
        "INSM": "INSMED Inc",
        "C": "CITIGROUP INC",
        "SPCX": "SPACE EXPLORATION TECHNOLOGIES CORP",
        "PHAR": "Pharming Group N.V.",
        "SLSN": "SOLESENCE, INC.",
        "GEV": "GE Vernova Inc.",
        "ORLY": "O REILLY AUTOMOTIVE INC",
        "GWW": "W.W. GRAINGER, INC.",
        "BBD": "BANK BRADESCO",
        "ON": "ON SEMICONDUCTOR CORP",
    }
)
NOBODY = "A summary that names nobody at all."


@pytest.mark.parametrize(
    "symbol, text, kind",
    [
        # Seen live in the cache: each of these three raised the buyout caution.
        ("SHOP", "Shopify Acquires SHOP.COM Domain Name", "acquisition"),
        ("UBER", "Uber Explores Sale of Delivery Hero Assets to Facilitate Takeover Approval", "acquisition"),
        ("SNDK", "SanDisk Advises Shareholders to Reject Unsolicited Mini-Tender Offer from Tutanota LLC", "disclosure"),
        # A full stop that ends a sentence still ends the name, and a sale of the company is still one.
        ("SHOP", "Tidewell to Acquire SHOP. Shares Jump", "takeover"),
        ("ACME", "Acme Explores Sale", "takeover"),
        ("ACME", "Acme Explores Sale of the Company After Approach from Tidewell", "takeover"),
        ("ACME", "Acme Explores Sale of Its Chip Unit", "acquisition"),
        # A list of names before the thing bought is not a list of companies bought.
        ("AAPL", "Longview Partners Acquires Apple, Microsoft, and NVIDIA Shares", "insider"),
        ("NVDA", "Tidewell Raises Bid for Nvidia and AMD Chips", "other"),
        ("INSM", "Sanofi to Acquire Insmed, Sending Shares Up 20%", "takeover"),
        ("INSM", "Sanofi to Acquire Insmed and Curevo in $45 Billion Deal", "takeover"),
    ],
)
def test_a_takeover_by_name_is_not_fooled_by_a_domain_a_list_or_a_sale_of_assets(symbol, text, kind):
    assert typed(symbol, text, NOBODY) == kind


@pytest.mark.parametrize(
    "symbol, text, kind",
    [
        # Seen live: the results rule took these, and results are neither pointed out nor studied.
        ("AMD", "AMD Provides Fourth Quarter 2026 Revenue Guidance", "guidance"),
        ("GME", "GameStop Announces Preliminary Second-Quarter 2026 Financial Results", "guidance"),
        ("CRWV", "CoreWeave Reports Weaker Second-Quarter Revenue Forecast", "guidance"),
        ("AVGO", "Broadcom Shares Decline as AI Revenue Outlook Fails to Exceed High Expectations", "guidance"),
        # Results are still results, with or without guidance alongside.
        ("ACME", "Acme Reports Third Quarter 2026 Financial Results", "earnings"),
        ("PFE", "Pfizer Reports Q2 Revenue Growth and Raises Full-Year Outlook", "earnings"),
        ("ACME", "Acme Announces Third Quarter 2026 Revenues Above Market Expectations", "earnings"),
        ("ACME", "Acme Reports Loss on Plant Closure but Lifts 2026 Financial Forecast", "earnings"),
        # "Target Price" is a price target, so the bank wrote the note.
        ("GS", "Goldman Sachs Raises Microsoft Target Price to $640", "another"),
        # Buying out of someone else's bankruptcy is not being in it.
        ("DIS", "Disney to Acquire Assets from Character.ai Following Bankruptcy Filing", "acquisition"),
        ("ACME", "Acme Files for Chapter 11 Bankruptcy Protection", "distress"),
        # Offerings named by their size or their kind.
        ("ACME", "Acme Announces Securities Purchase Agreement for $5.4 Million Offering", "financing"),
        ("ACME", "Acme Announces Pricing of $2.5 Million Underwritten Offering", "financing"),
        ("ACME", "Acme Files for Potential $100 Million Securities Offering", "financing"),
        # A bank managing someone else's offering is doing business, and a credit default swap is no default.
        ("C", "Citi Acts as Underwriter for Amphenol's $1.1 Billion Senior Notes Offering", "agreement"),
        ("SPCX", "SpaceX 5-Year Credit Default Swaps Widen", "other"),
        # Someone else after the name, with or without an apostrophe.
        ("AAPL", "Apple's Supplier Foxconn Cuts Outlook", "other"),
        ("AAPL", "Apple's Top Chip Supplier Cuts Full-Year Revenue Outlook", "other"),
        ("AAPL", "Apple's Services Unit Raises Full-Year Revenue Outlook", "guidance"),
    ],
)
def test_rules_the_second_review_found_too_loose_or_too_tight(symbol, text, kind):
    assert typed(symbol, text, NOBODY) == kind


def test_a_notice_filed_with_the_results_keeps_its_own_type():
    def typed_as(items):
        found = events.classify(filing(items))
        return found.event, found.title

    # Seen live in ten filings: a delisting notice, a restatement or a new auditor, filed with the quarter's results.
    assert typed_as(["2.02", "3.01", "9.01"]) == ("distress", "8-K: Delisting notice, or a listing rule not met; Results of operations")
    assert typed_as(["2.02", "4.02"])[0] == "distress" and typed_as(["2.02", "4.01", "5.02"])[0] == "distress"
    assert typed_as(["2.02", "5.01"])[0] == "takeover"
    # Results still come ahead of everything routine.
    assert typed_as(["2.02", "7.01", "9.01"])[0] == "earnings" and typed_as(["5.02", "2.02", "1.01"])[0] == "earnings"
    codes = list(events.FILING_ITEMS)
    flagged = [code for code in codes if events.FILING_ITEMS[code][0] in ("distress", "takeover")]
    assert all(codes.index(code) < codes.index("2.02") for code in flagged) and len(flagged) == 6


@pytest.mark.parametrize(
    "symbol, text, named",
    [
        # Seen live: each of these companies had most of its own headlines set aside.
        ("SPCX", "SpaceX to Launch Starship Flight 14", True),  # a brand, listed for it
        ("C", "Citi Launches New Card for Small Businesses", True),
        ("SLSN", "Sol\u00e9sence, Inc. Reports Third Quarter 2026 Financial Results", True),  # an accent
        ("BBD", "Bradesco Launches New Marketing Campaign", True),  # the telling word after a common one
        ("GEV", "GE Vernova Wins Turbine Order", True),  # a two-letter first word is not the whole name
        ("GEV", "Vernova Wins Turbine Order", True),
        ("ORLY", "O'Reilly Automotive Opens 200th Store", True),
        ("GWW", "W.W. Grainger Opens Distribution Centre", True),
        ("GWW", "Grainger Opens Distribution Centre", True),
        # Other companies whose ticker in London is the same, also seen live: still another company's news.
        ("COST", "Costain Group PLC Announces Total Voting Rights", False),
        ("PHAR", "Pharos Energy plc: Form 8.5 (EPT/RI) Disclosure", False),
        ("C", "Citizens Financial Opens New Branch", False),
        ("ON", "Tidewell Semiconductor Opens New Fab", False),  # a plain word names nobody
    ],
)
def test_what_counts_as_naming_the_company(symbol, text, named):
    assert (typed(symbol, text, NOBODY) != "another") is named


def test_how_much_of_a_name_is_needed_when_it_opens_with_a_letter_or_two():
    assert events._needed(["ge", "vernova"]) == 2 and events._needed(["w", "w", "grainger"]) == 3
    assert events._needed(["o", "reilly", "automotive"]) == 2 and events._needed(["at&t"]) == 1
    assert events._needed(["x"]) is None and events._needed(["a", "b"]) is None and events._needed(["bank", "of"]) is None
    # Seen live: 279 of the 10,400 companies the SEC lists had no usable name for this reason.
    for symbol in ("GEV", "ORLY", "GWW", "ON"):
        assert events._named(events.aliases(symbol, COMPANIES[symbol])) is not None
    # The by-name takeover rule gets the same names, and spells them strictly.
    assert typed("GWW", "Tidewell to Acquire W.W. Grainger for $60 Billion") == "takeover"
    assert typed("GEV", "Tidewell to Acquire GE Vernova in All-Stock Deal") == "takeover"
    assert typed("GEV", "Tidewell to Acquire GE Appliances Unit") != "takeover"


def test_a_company_also_goes_by_the_names_it_had_and_the_brands_listed_for_it():
    names = events.aliases("SRTM", "Stratum Holdings Corp.")
    also = events.also_called(("La Rosa Holdings Corp.", "La Rosa Holdings Corp.", "", "Sol\u00e9sence, Inc."))
    assert names == ("SRTM", "stratum holdings")
    assert also == ("la rosa holdings", "solesence")  # cleaned like a name, accents off, and none twice
    old = "La Rosa Holdings Launches 'JAEME' AI Technology System"
    assert events.headline_type(old, names, NOBODY, also) == "product"
    assert events.headline_type(old, names, NOBODY) == "another"

    nubank = events.aliases("NU", "Nu Holdings Ltd."), events.also_called(("Nubank",))
    assert nubank == (("nu holdings",), ("nubank",))
    assert events.headline_type("Nubank Raises Full-Year Revenue Guidance", nubank[0], NOBODY, nubank[1]) == "guidance"

    # A service with nowhere to look former names up has none, and a lookup that fails gives none.
    def down(symbol, since):
        raise httpx.ConnectError("down")

    since = date(2023, 8, 1)
    assert NewsService(NullCache()).former_names("ACME", since) == ()
    assert NewsService(NullCache(), former_names=down).former_names("ACME", since) == ()
    asked = []
    service = NewsService(NullCache(), former_names=lambda symbol, day: asked.append((symbol, day)) or ["Old Acme Inc"])
    assert service.former_names("ACME", since) == ("Old Acme Inc",) and asked == [("ACME", since)]


@pytest.mark.parametrize(
    "given, plain",
    [
        ("COSTCO WHOLESALE CORP /NEW", "costco wholesale"),  # what the SEC adds to tell registrants apart
        ("BANK OF AMERICA CORP /DE/", "bank of america"),
        ("BANKAMERICA CORP/DE/", "bankamerica"),
        ("APPLE COMPUTER INC/ FA", "apple computer"),
        ("M/I HOMES, INC.", "m i homes"),  # a slash inside a name is part of it
        ("PRICE/COSTCO INC", "price costco"),  # seen live: this had become the alias "price"
        ("20/20 Biolabs, Inc.", "20 20 biolabs"),
        ("The Cigna Group", "cigna group"),
        ("Sol\u00e9sence, Inc.", "solesence"),
        ("", ""),
        ("/DE/", ""),
    ],
)
def test_a_registered_name_is_cleaned_without_losing_what_follows_a_slash(given, plain):
    assert events._plain_name(given) == plain
    assert events.aliases("ZZZZ", given) == (("ZZZZ", plain) if plain else ("ZZZZ",))


def test_a_name_with_a_slash_in_it_is_found_as_headlines_write_it():
    names = events.aliases("MHO", "M/I HOMES, INC.")
    assert names == ("MHO", "m i homes") and events._named(names) is not None
    assert events.headline_type("M/I Homes Opens New Community in Tampa", names, NOBODY) == "product"
    assert events.headline_type("Tidewell to Acquire M/I Homes for $4 Billion", names, NOBODY) == "takeover"
    assert events.headline_type("Tidewell Homes Opens New Community", names, NOBODY) == "another"  # "homes" names nobody


@pytest.mark.parametrize(
    "symbol, name, text",
    [
        # Seen live in the cache after a word built on the name was allowed to stand for it.
        ("META", "Meta Platforms, Inc.", "Inderes Lowers Price Target for Metacon"),
        ("META", "Meta Platforms, Inc.", "Metacon Awarded EU Funding for Hydrogen Project"),
        ("BAC", "BANK OF AMERICA CORP /DE/", "Bank of England Cuts 2026 Growth Forecast"),
        ("BAC", "BANK OF AMERICA CORP /DE/", "Bank of Japan Raises Inflation Outlook"),
        ("GS", "GOLDMAN SACHS GROUP INC", "Gold Miner Raises Full-Year Production Outlook"),
        ("INTC", "INTEL CORP", "Intelligence Firm Raises Full-Year Revenue Guidance"),
        ("CRM", "Salesforce, Inc.", "Sales Forecast Raised at Tidewell Systems"),
        ("COST", "COSTCO WHOLESALE CORP /NEW", "Cost of Living Forecast Raised by Tidewell Research"),
        ("SHOP", "SHOPIFY INC.", "Shop Owners Raise Full-Year Sales Outlook"),
        ("GME", "GameStop Corp.", "Game Maker Raises Full-Year Revenue Guidance"),
        ("COIN", "Coinbase Global, Inc.", "Coin Issuer Prices $500 Million Notes Offering"),
    ],
)
def test_a_word_that_only_starts_like_the_name_does_not_name_the_company(symbol, name, text):
    names = events.aliases(symbol, name)
    assert events.headline_type(text, names, NOBODY) == "another"
    # In particular it is not the company opening the headline, which would turn a flagged type on.
    assert events._named(names)[0].match(text) is None


@pytest.mark.parametrize(
    "symbol, text",
    [
        # Seen live: a brand that is one unit of the company, and a name it had decades ago.
        ("C", "Banorte to Acquire Citibank for $5 Billion"),
        ("META", "Tidewell to Acquire WhatsApp for $30 Billion"),
        ("NU", "Tidewell to Acquire Nubank for $40 Billion"),
    ],
)
def test_a_takeover_by_name_goes_by_the_companys_own_name_and_no_other(symbol, text):
    COMPANIES.setdefault("META", "Meta Platforms, Inc.")
    COMPANIES.setdefault("NU", "Nu Holdings Ltd.")
    kind = typed(symbol, text, NOBODY)
    # The headline is about the company, so it is not set aside. But no buyout caution is raised on it.
    assert kind not in ("takeover", "another")
    # Under its registered name it is a takeover as before.
    assert events.headline_type("Tidewell to Acquire Citigroup for $200 Billion", events.aliases("C", COMPANIES["C"]), NOBODY) == "takeover"


@pytest.mark.parametrize(
    "symbol, text",
    [
        # Seen live for every company with a name of two words or more: the check passed on the name's own second word.
        ("COST", "Costco Wholesale Supplier Foxconn Cuts Full-Year Revenue Outlook"),
        ("COST", "Costco Wholesale's Supplier Foxconn Cuts Full-Year Revenue Outlook"),
        ("COST", "Costco Supplier Foxconn Cuts Full-Year Revenue Outlook"),
        ("GS", "Goldman Sachs Partner Firm Cuts Full-Year Revenue Outlook"),
        ("JPM", "JPMorgan Chase Rival Bank Cuts Full-Year Revenue Outlook"),
        ("PLTR", "Palantir Technologies Customer Tidewell Seeks $1 Billion Financing"),
        ("AAPL", "Apple's Top Chip Supplier Cuts Full-Year Revenue Outlook"),
        # A legal ending between the name and what follows does not hide it.
        ("AAPL", "Apple Inc.'s Supplier Foxconn Cuts Full-Year Revenue Outlook"),
        ("AAPL", "Apple Inc. Supplier Foxconn Cuts Full-Year Revenue Outlook"),
        ("JPM", "JPMorgan Chase & Co. Rival Bank Cuts Full-Year Revenue Outlook"),
    ],
)
def test_someone_else_after_the_name_is_seen_however_long_the_name_is(symbol, text):
    names = events.aliases(symbol, COMPANIES[symbol])
    opens = events._named(names)[0]
    lead = opens.match(text)
    assert lead is not None and events._SOMEONE_ELSE.match(text, lead.end())
    assert events.headline_type(text, names, NOBODY) not in events.IN_FRONT
    # The company's own news, under the same long name, still counts.
    own = text.split(" Supplier ")[0].split("'s ")[0].split(" Partner ")[0].split(" Rival ")[0].split(" Customer ")[0]
    assert events.headline_type(own + " Raises Full-Year Revenue Guidance", names, NOBODY) == "guidance"
    assert events.headline_type(own + "'s Services Unit Raises Full-Year Revenue Guidance", names, NOBODY) == "guidance"


def test_another_name_is_given_whole():
    # Seen live: SunPower was Complete Solaria until this year, and "Complete" alone is anybody.
    names, also = events.aliases("SPWR", "SunPower Inc."), events.also_called(("Complete Solaria, Inc.",))
    assert also == ("complete solaria",)
    opens, _ = events._named(names, also)
    other = "Complete Energy Announces $50 Million Share Offering"
    assert events.headline_type(other, names, NOBODY, also) == "another" and opens.match(other) is None
    assert events.headline_type("Complete Solaria Announces $50 Million Share Offering", names, NOBODY, also) == "financing"
    assert events.headline_type("SunPower Announces $50 Million Share Offering", names, NOBODY, also) == "financing"
    # A brand of one word is the whole of it, and the company's own name still goes by its first word.
    assert events._named(events.aliases("C", "CITIGROUP INC"), ("citi",))[0].match("Citi Launches New Card")
    assert events._named(events.aliases("PXD", "PIONEER NATURAL RESOURCES CO"))[0].match("Pioneer Raises Guidance")
    assert events._named((), ("k wave media",))[0].match("K Wave Media Executes 1-for-30 Reverse Stock Split")
    assert events._named((), ("k wave media",))[0].match("K Street Lobbyists Gather") is None


def test_a_plain_word_of_a_name_names_nobody():
    rocket = events.aliases("RKT", "Rocket Companies, Inc.")
    # Seen live: "companies" stood for Rocket Companies, and let two of Reckitt's headlines through.
    assert events.headline_type("Reckitt Named Among Best Companies to Work for in Asia", rocket, "Reckitt was named.") == "another"
    assert events.headline_type("Rocket Companies Named Among Best Places to Work", rocket, NOBODY) != "another"
    assert {"companies", "company", "sales", "games", "homes"} <= events._PLAIN_WORDS


def test_every_rule_points_at_a_type_with_a_label():
    known = set(events.EVENT_TYPES)
    assert {kind for kind, _ in events.HEADLINE_RULES} <= known
    assert {kind for kind, _ in events.FILING_ITEMS.values()} <= known
    assert events.NOT_STUDIED <= known and events.FLAGGED <= known and set(events.CAUTIONS) <= events.FLAGGED
    assert not events.FLAGGED & events.NOT_STUDIED


def filing(items, when=datetime(2026, 11, 20, 16, 30, tzinfo=ET), form="8-K", ref="f1"):
    return NewsItem(when=when, kind="filing", title=form, source="SEC EDGAR", ref=ref, items=tuple(items))


NOON = datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc)


def headline(n: int, when: datetime, title: str = "Acme Launches Next-Generation Platform") -> NewsItem:
    return NewsItem(when=when, kind="headline", title=title, source="fiscal.ai", url=f"https://example.com/{n}", ref=str(n))


def test_a_filing_takes_its_type_from_its_most_telling_item():
    both = events.classify(filing(["7.01", "5.02", "9.01"]))
    assert both.event == "management"
    assert both.title == "8-K: Departure or appointment of directors or officers; Regulation FD disclosure"

    # Results come with other items attached, and it is still an earnings report.
    assert events.classify(filing(["2.02", "7.01", "9.01"])).event == "earnings"
    assert events.classify(filing(["8.01", "1.03"])).event == "distress"

    amended = events.classify(filing(["5.02"], form="8-K/A"))
    assert amended.title == "8-K/A: Departure or appointment of directors or officers"

    # Exhibits alone, or a code the table does not know, say nothing about the event.
    paperwork = events.classify(filing(["9.01", "6.02"]))
    assert (paperwork.event, paperwork.title) == ("other", "8-K")


# -- What followed ----------------------------------------------------------------


def sessions(count: int, start: date = date(2026, 6, 1)) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def bars_from(returns: list[float], start: date = date(2026, 6, 1)) -> list[Bar]:
    """Closes on consecutive weekdays: 100 on `start`, then one return a session."""
    closes = [100.0]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return [Bar(day, close) for day, close in zip(sessions(len(closes), start), closes)]


def news(day: date, at: clock, kind: str, title: str = "headline", ref: str = "") -> NewsItem:
    when = datetime.combine(day, at, tzinfo=ET)
    return NewsItem(when=when, kind="headline", title=title, source="test", ref=ref or f"{when}{title}", event=kind)


def test_an_event_is_measured_on_the_first_session_that_reflects_it():
    #        Mon Jun 1 (100), Tue +1%, Wed +2%, Thu -3%, Fri +4%, Mon -5%, then flat
    bars = bars_from([0.01, 0.02, -0.03, 0.04, -0.05] + [0.0] * 10)
    wed, sat = date(2026, 6, 3), date(2026, 6, 6)
    items = [
        news(wed, clock(10, 0), "guidance"),  # during the session: Wednesday's own move
        news(wed, clock(16, 0), "legal"),  # at the close or after it: Thursday's
        news(sat, clock(9, 0), "management"),  # at a weekend: Friday's close to Monday's
        # The same instant as the second one, given in UTC as a feed does.
        NewsItem(datetime(2026, 6, 3, 20, 0, tzinfo=timezone.utc), "headline", "utc", "test", ref="u", event="drug"),
    ]
    found = events.study(items, bars, set(), ["guidance", "legal", "management", "drug"])
    moves = {kind: study.moves[0] for kind, study in found.items()}
    assert (moves["guidance"].day, moves["guidance"].move_pct) == (wed, pytest.approx(0.02))
    assert (moves["legal"].day, moves["legal"].move_pct) == (date(2026, 6, 4), pytest.approx(-0.03))
    assert (moves["management"].day, moves["management"].move_pct) == (date(2026, 6, 8), pytest.approx(-0.05))
    assert (moves["drug"].day, moves["drug"].move_pct) == (date(2026, 6, 4), pytest.approx(-0.03))
    # What followed: that session's close to the close five sessions later.
    assert moves["guidance"].after_pct == pytest.approx(0.97 * 1.04 * 0.95 - 1)


def test_event_days_are_set_beside_the_stocks_usual_day():
    # Most days move 1% either way. Four event days move 5%.
    returns = [0.01 if n % 2 else -0.01 for n in range(100)]
    for n, size in ((20, 0.05), (40, -0.05), (60, 0.05), (80, 0.05)):
        returns[n - 1] = size  # returns[n - 1] is the move on session n
    bars = bars_from(returns)
    days = [b.day for b in bars]
    items = [news(days[n], clock(10, 0), "guidance", f"event {n}") for n in (20, 40, 60, 80)]

    study = events.study(items, bars, set(), ["guidance"])["guidance"]
    assert (study.event, study.label) == ("guidance", "Guidance or outlook")
    assert (study.count, study.up, study.down, study.since) == (4, 3, 1, days[20])
    assert study.mean_abs == pytest.approx(0.05)
    # Every session from the first event on, the event days among them: 4 of 5% and 77 of 1%.
    assert study.ordinary_abs == pytest.approx((4 * 0.05 + 77 * 0.01) / 81)
    assert [m.title for m in study.moves] == ["event 80", "event 60", "event 40", "event 20"]  # newest first

    closes = [b.close for b in bars]
    assert study.after_count == 4
    assert study.mean_abs_after == pytest.approx(sum(abs(closes[n + 5] / closes[n] - 1) for n in (20, 40, 60, 80)) / 4)
    # Every five-session span from the first event on, whether or not it holds an event.
    spans = [abs(closes[i + 5] / closes[i] - 1) for i in range(20, len(closes) - 5)]
    assert study.ordinary_abs_after == pytest.approx(sum(spans) / len(spans))


def test_a_day_that_carries_an_earnings_report_is_left_out():
    returns = [0.01 if n % 2 else -0.01 for n in range(100)]
    returns[19], returns[39], returns[59] = 0.05, -0.05, 0.20  # sessions 20 and 40, and a report on 60
    bars = bars_from(returns)
    days = [b.day for b in bars]
    items = [news(days[n], clock(10, 0), "guidance") for n in (20, 40, 60)]

    study = events.study(items, bars, {days[60]}, ["guidance"])["guidance"]
    # The 20% belongs to the report: it is neither an event of this type nor part of the usual day.
    assert study.count == 2 and study.mean_abs == pytest.approx(0.05)
    assert study.ordinary_abs == pytest.approx((2 * 0.05 + 78 * 0.01) / 80)
    assert days[60] not in {m.day for m in study.moves}


def test_a_stretch_with_an_earnings_report_in_it_is_left_out_too():
    # Quiet but for a 20% report on session 60. One event comes three sessions before it.
    returns = [0.01 if n % 2 else -0.01 for n in range(100)]
    returns[59] = 0.20
    bars = bars_from(returns)
    days, closes = [b.day for b in bars], [b.close for b in bars]
    items = [news(days[n], clock(10, 0), "guidance", f"event {n}") for n in (20, 30, 40, 57)]

    study = events.study(items, bars, {days[60]}, ["guidance"])["guidance"]
    after = {m.title: m.after_pct for m in study.moves}
    # The event counts for its own day. What followed it was the report, so nothing is said of that.
    assert study.count == 4 and after["event 57"] is None and after["event 40"] is not None
    assert study.after_count == 3
    assert study.mean_abs_after == pytest.approx(sum(abs(closes[n + 5] / closes[n] - 1) for n in (20, 30, 40)) / 3)
    # The usual five sessions leave out the five spans that take in session 60.
    spans = [abs(closes[i + 5] / closes[i] - 1) for i in range(20, len(closes) - 5) if not i < 60 <= i + 5]
    assert len(spans) == len(closes) - 5 - 20 - 5
    assert study.ordinary_abs_after == pytest.approx(sum(spans) / len(spans))
    assert study.ordinary_abs_after < 0.011  # the report would have made it more than twice that


def test_several_items_on_one_session_are_one_event():
    bars = bars_from([0.01] * 30)
    days = [b.day for b in bars]
    items = [
        news(days[9], clock(17, 0), "analyst", "the evening before"),  # after the close: felt on session 10
        news(days[10], clock(8, 0), "analyst", "before the open"),
        news(days[10], clock(11, 0), "analyst", "mid-morning"),
        news(days[10], clock(11, 0), "legal", "another type the same day"),
    ]
    found = events.study(items, bars, set(), ["analyst", "legal"])
    assert found["analyst"].count == 1 and found["analyst"].moves[0].title == "the evening before"
    assert found["legal"].count == 1


def test_an_event_is_counted_only_once_the_session_that_reflects_it_has_closed():
    bars = bars_from([0.01] * 30)
    days = [b.day for b in bars]
    items = [
        news(days[-1], clock(16, 30), "guidance", "after the last close on record"),
        news(days[-1] + timedelta(days=1), clock(9, 0), "guidance", "the next morning"),
        news(days[-3], clock(10, 0), "guidance", "three sessions ago"),
    ]
    study = events.study(items, bars, set(), ["guidance"])["guidance"]
    assert [m.title for m in study.moves] == ["three sessions ago"]
    # Too recent for five sessions to have followed, so nothing is said about what followed.
    assert study.moves[0].after_pct is None
    assert (study.after_count, study.mean_abs_after, study.ordinary_abs_after) == (0, None, None)

    # With nothing but unfinished events there is no study at all.
    assert events.study(items[:2], bars, set(), ["guidance"]) == {}


def test_the_usual_day_is_measured_over_at_least_a_quarter():
    # Two events in the last week, which was wild. Before it: quiet sessions, and wilder ones earlier still.
    returns = [0.10] * 40 + [0.01] * 55 + [0.04] * 5
    bars = bars_from(returns)
    days = [b.day for b in bars]
    items = [news(days[-4], clock(10, 0), "legal"), news(days[-2], clock(10, 0), "legal")]
    study = events.study(items, bars, set(), ["legal"])["legal"]
    assert study.mean_abs == pytest.approx(0.04)
    # Not just the days since the first event (all 4%), and not the stretch before the last 60 sessions.
    assert events.BASELINE_SESSIONS == 60
    assert study.ordinary_abs == pytest.approx((55 * 0.01 + 5 * 0.04) / 60)


def test_types_that_are_not_events_get_no_study():
    bars = bars_from([0.01] * 30)
    day = bars[10].day
    items = [news(day, clock(10, 0), kind) for kind in ("earnings", "calendar", "other")]
    assert events.study(items, bars, set(), ["earnings", "calendar", "other", "guidance"]) == {}


def test_a_hole_in_the_price_history_drops_the_event():
    early, late = bars_from([0.01] * 5, date(2026, 6, 1)), bars_from([0.01] * 5, date(2026, 7, 1))
    in_the_hole = news(date(2026, 6, 17), clock(10, 0), "legal")
    assert events.study([in_the_hole], early + late, set(), ["legal"]) == {}


def test_context_lists_recent_news_and_studies_only_the_types_in_it():
    bars = bars_from([0.01] * 120, date(2026, 6, 1))
    days = [b.day for b in bars]
    last = days[-1]
    today = last + timedelta(days=1)

    def raw(day, title, n):
        return NewsItem(datetime.combine(day, clock(10, 0), tzinfo=ET), "headline", title, "test", ref=str(n))

    recent = [raw(last - timedelta(days=n % 5), f"Acme Raises Full-Year Revenue Guidance {n}", n) for n in range(12)]
    older = [
        raw(days[30], "Acme Raises Full-Year Revenue Guidance", 100),
        raw(days[50], "Acme Faces Class Action Lawsuit", 101),  # a type with nothing recent
        raw(today - timedelta(days=events.RECENT_DAYS + 1), "Acme Launches Next-Generation Platform", 102),  # a day too old
    ]
    extra = [raw(last, "Acme to Present at Lakeshore Technology Conference", 200), filing(["5.02"], when=datetime.combine(last, clock(8, 0), tzinfo=ET))]

    shown, counts, studies, elsewhere = events.context(recent + older + extra, bars, set(), today)[:4]
    assert elsewhere == []
    assert len(shown) == events.MAX_RECENT and shown == sorted(shown, key=lambda i: i.when, reverse=True)
    assert all(item.event for item in shown)  # classified on the way through
    # Every recent item is counted, not only the ones listed, in the order the types are defined.
    assert counts == {"guidance": 12, "management": 1, "calendar": 1}
    # A study draws on the whole record, and there is none for a type with nothing recent.
    assert set(studies) == {"guidance", "management"}
    assert studies["guidance"].since == days[30]


def test_a_flagged_item_stays_on_the_list_when_a_busy_week_pushes_it_out():
    bars = bars_from([0.01] * 120, date(2026, 6, 1))
    last = bars[-1].day
    today = last + timedelta(days=1)

    def raw(day, hour, title, n):
        return NewsItem(datetime.combine(day, clock(hour, 0), tzinfo=ET), "headline", title, "test", ref=str(n))

    launches = [raw(last, 9 + n % 6, f"Acme Launches Product {n}", n) for n in range(12)]
    older = [
        raw(last - timedelta(days=9), 10, "Acme Prices $300 Million Offering of Convertible Notes", 100),
        raw(last - timedelta(days=10), 10, "Acme Raises Full-Year Revenue Guidance", 101),
        raw(last - timedelta(days=11), 10, "Acme Raises Full-Year Revenue Guidance", 102),  # only the newest of a type is kept
        raw(last - timedelta(days=12), 10, "Acme Appoints New Chief Financial Officer", 103),  # not a flagged type
    ]
    shown, counts, _, _ = events.context(launches + older, bars, set(), today)[:4]
    assert len(shown) == events.MAX_RECENT + 2 and shown == sorted(shown, key=lambda i: i.when, reverse=True)
    assert [i.ref for i in shown[-2:]] == ["100", "101"]
    assert counts == {"guidance": 2, "financing": 1, "management": 1, "product": 12}


def test_another_companys_news_takes_no_place_and_counts_towards_nothing():
    bars = bars_from([0.01] * 120, date(2026, 6, 1))
    days = [b.day for b in bars]
    last = days[-1]
    today = last + timedelta(days=1)
    names = events.aliases("ACME", "Acme")

    def raw(day, hour, title, n, summary=""):
        when = datetime.combine(day, clock(hour, 0), tzinfo=ET)
        return NewsItem(when, "headline", title, "test", ref=str(n), summary=summary)

    # A week of the company holding other companies' shares, newer than anything of its own.
    holdings = [raw(last, 9 + n % 6, f"Acme Updates Major Holdings in Company {n} PLC", n) for n in range(14)]
    lookalike = raw(last, 15, "Acmain Group PLC Announces Total Voting Rights", 50, "Acmain Group PLC reported its voting rights.")
    own = [
        raw(last - timedelta(days=2), 10, "Acme Raises Full-Year Revenue Guidance", 100),
        raw(last - timedelta(days=3), 10, "Longview Partners Increases Stake in Acme to 6.8%", 101),
    ]
    # Older ones, for the studies: an insider trade in the company, and holdings of its own in others.
    older = [raw(days[40], 10, "Acme Director Sells Shares Worth $2 Million", 200)] + [
        raw(days[50 + n], 10, f"Acme Increases Stake in Company {n} PLC", 300 + n) for n in range(5)
    ]

    shown, counts, studies, elsewhere = events.context(holdings + [lookalike] + own + older, bars, set(), today, names)[:4]
    # The company's own two items are listed, though fifteen newer ones were tagged with it.
    assert [i.ref for i in shown] == ["100", "101"]
    assert counts == {"guidance": 1, "insider": 1, "another": 15}
    # The newest of the rest are handed back apart, to be looked at.
    assert len(elsewhere) == events.MAX_ELSEWHERE == 10 and all(i.event == "another" for i in elsewhere)
    assert elsewhere[0].ref == "50" and elsewhere == sorted(elsewhere, key=lambda i: i.when, reverse=True)
    # The insider study rests on trades in this company's shares: the two of them, not its five holdings.
    assert set(studies) == {"guidance", "insider"} and studies["insider"].count == 2
    assert not any(i.event == "another" for i in shown)

    # Without the company's name, every one of them is typed as it always was.
    plain = events.context(holdings + [lookalike] + own + older, bars, set(), today)
    assert plain[3] == [] and "another" not in plain[1] and plain[2]["insider"].count > 2


def test_a_short_list_always_has_the_headline_behind_a_caution():
    day = date(2026, 11, 20)
    deal, split = news(day, clock(8, 0), "takeover", "the deal"), news(day, clock(7, 0), "split", "the split")
    kinds = ["legal", "analyst", "legal", "guidance", "guidance", "guidance"]
    newer = [news(day, clock(10 + n, 0), kind, f"{kind} at {10 + n}") for n, kind in enumerate(kinds)]

    # Three newer items of other types: the deal keeps a place, and the list is still newest first.
    assert events.shortlist([deal, *newer[:3]], 3) == [newer[2], newer[1], deal]
    # A run of another flagged type cannot crowd it out either: one of each flagged type goes in first.
    assert events.shortlist([deal, *newer], 3) == [newer[5], newer[4], deal]
    assert events.shortlist([deal, split, *newer], 6) == [newer[5], newer[4], newer[3], newer[2], deal, split]
    # With less room than flagged types, the ones that carry a caution come first.
    assert set(events.CAUTIONS) == {"takeover", "split"}
    assert events.shortlist([deal, split, *newer], 2) == [deal, split]
    # With nothing flagged it is simply the newest.
    assert events.shortlist(newer[:3], 2) == [newer[2], newer[1]]
    assert events.shortlist([], 3) == []


# -- Alpaca headlines -------------------------------------------------------------


def alpaca(handler) -> AlpacaData:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return AlpacaData("key-id", "secret", client=client, sleep=lambda _: None)


def test_headlines_parse_a_captured_response_and_follow_its_pages():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=NEWS["second" if "page_token" in request.url.params else "first"])

    items = alpaca(handler).headlines("ACME", datetime(2026, 11, 2, tzinfo=ET))

    assert seen[0].url.path == "/v1beta1/news"
    assert dict(seen[0].url.params) == {
        "symbols": "ACME",
        "start": "2026-11-02T05:00:00Z",  # midnight in New York, sent in UTC
        "limit": "50",
        "sort": "desc",
        "include_content": "false",
    }
    assert seen[1].url.params["page_token"] == NEWS["first"]["next_page_token"] and len(seen) == 2
    assert len(items) == 6 and items == sorted(items, key=lambda i: i.when, reverse=True)
    assert items[0] == NewsItem(
        when=datetime(2026, 11, 24, 19, 2, 45, tzinfo=timezone.utc),
        kind="headline",
        title="Acme Director Sells Shares Worth $2 Million",
        source="fiscal.ai",
        url="https://news.example.com/acme/director-share-sale?source=feed&campaign=1",
        ref="129880987",
        # The feed's one sentence about the story, kept to tell whose news it is.
        summary="A director of Acme reported selling shares worth about $2 million, according to a regulatory filing.",
    )
    # Two stories can share a time stamp. Their ids keep them apart.
    assert items[2].when == items[3].when and items[2].ref != items[3].ref
    assert [events.classify(i).event for i in items] == ["insider", "agreement", "analyst", "product", "guidance", "other"]


def test_headline_rows_that_cannot_be_used_are_skipped_and_only_web_links_are_kept():
    rows = [
        {"id": 1, "headline": "  Acme   Opens\nNew Office ", "created_at": "2026-11-20T15:00:00Z", "url": "javascript:alert(1)"},
        {"id": 2, "headline": "", "created_at": "2026-11-20T15:00:00Z"},
        {"id": 3, "headline": "No time", "created_at": None},
        {"id": 4, "headline": "No zone", "created_at": "2026-11-20T15:00:00"},
        "not a row",
        {"id": 5, "headline": "Plain link", "created_at": "2026-11-19T15:00:00Z", "url": "HTTPS://example.com/a"},
        {"id": 6, "headline": "With a summary", "created_at": "2026-11-18T15:00:00Z", "summary": "  Acme  said\tso. " + "x" * 900},
        {"id": 7, "headline": "Summary of the wrong kind", "created_at": "2026-11-17T15:00:00Z", "summary": None},
    ]
    items = alpaca(lambda request: httpx.Response(200, json={"news": rows, "next_page_token": None})).headlines("ACME", datetime(2026, 11, 1, tzinfo=ET))
    assert [(i.ref, i.title, i.url, i.source) for i in items[:2]] == [
        ("1", "Acme Opens New Office", None, "Alpaca"),
        ("5", "Plain link", "HTTPS://example.com/a", "Alpaca"),
    ]
    # The summary is a third party's text too: tidied, and cut to a fixed length.
    assert [i.summary for i in items[:2]] == ["", ""] and items[3].summary == ""
    assert items[2].summary.startswith("Acme said so. xxx") and len(items[2].summary) == SUMMARY_CHARS == 400


def test_a_very_long_run_of_headlines_is_cut_off_and_the_scan_says_so():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        row = {"id": len(calls), "headline": "More", "created_at": "2026-11-20T15:00:00Z"}
        return httpx.Response(200, json={"news": [row], "next_page_token": "more"})

    market = alpaca(handler)
    assert len(market.headlines("ACME", datetime(2026, 1, 1, tzinfo=ET))) == MAX_NEWS_PAGES == len(calls)
    assert market.notes == ["ACME: only the newest 2,000 headlines were loaded."]


# -- SEC filings ------------------------------------------------------------------


def edgar(tmp_path, recent: dict) -> EdgarEarnings:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files/company_tickers.json":
            return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
        return httpx.Response(200, json={"filings": {"recent": recent, "files": []}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EdgarEarnings("Test Person test@example.com", DiskCache(tmp_path), client=client, sleep=lambda _: None)


def test_edgar_lists_each_8k_with_its_items_and_a_link(tmp_path):
    recent = {
        "form": ["8-K", "10-Q", "8-K/A", "4", "8-K"],
        "items": ["1.01,7.01,9.01", "", "5.02", "", "2.02,9.01"],
        "acceptanceDateTime": [
            "2026-11-24T11:04:08.000Z",
            "2026-11-05T21:10:00.000Z",
            "2026-09-01T20:30:35.000Z",
            "2026-08-03T22:00:00.000Z",
            "2026-07-30T20:30:40.000Z",
        ],
        "accessionNumber": ["0000320193-26-000105", "0000320193-26-000104", "0000320193-26-000103", "0000320193-26-000102", ""],
        "primaryDocument": ["agreement_8k.htm", "q3_10q.htm", "officers_8ka.htm", "form4.xml", "results_8k.htm"],
    }
    found = edgar(tmp_path, recent).filings("AAPL")

    assert [(f.title, f.items) for f in found] == [
        ("8-K", ("1.01", "7.01", "9.01")),
        ("8-K/A", ("5.02",)),
        ("8-K", ("2.02", "9.01")),
    ]
    first = found[0]
    assert first.when == datetime(2026, 11, 24, 6, 4, 8, tzinfo=ET) and (first.kind, first.source) == ("filing", "SEC EDGAR")
    assert first.url == "https://www.sec.gov/Archives/edgar/data/320193/000032019326000105/agreement_8k.htm"
    assert first.ref == "0000320193-26-000105"
    assert found[2].url is None  # no accession number, so nothing to link to
    assert [events.classify(f).event for f in found] == ["agreement", "management", "earnings"]


def test_edgar_filings_from_a_real_filing_index(tmp_path):
    recent = json.loads((FIXTURES / "edgar_aapl.json").read_text(encoding="utf-8"))["filings"]["recent"]
    found = edgar(tmp_path, recent).filings("AAPL")
    assert len(found) == 27 and found == sorted(found, key=lambda f: f.when, reverse=True)
    kinds = [events.classify(f).event for f in found]
    assert kinds.count("earnings") == 12 and kinds.count("management") == 7
    # This capture carries no document names, so each link goes to the filing's folder.
    assert all(f.url.startswith("https://www.sec.gov/Archives/edgar/data/320193/") and f.url.endswith("/") for f in found)


def test_edgar_knows_a_company_by_name(tmp_path):
    source = edgar(tmp_path, {})
    assert source.company_name("AAPL") == "Apple Inc." and source.company_name("ZZZZ") is None
    assert source._cik("AAPL") == 320193  # the same table still finds a company's filings
    # The names came with the list of tickers, which is on disk for the next run.
    again = EdgarEarnings("Test Person test@example.com", DiskCache(tmp_path), client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    assert again.company_name("AAPL") == "Apple Inc."


def test_edgar_tells_a_foreign_filer_from_a_company_with_no_8k(tmp_path):
    foreign = {"form": ["6-K", "20-F", "6-K"], "items": ["", "", ""], "acceptanceDateTime": ["2026-11-01T12:00:00.000Z"] * 3}
    with pytest.raises(Unsupported, match="Form 6-K"):
        edgar(tmp_path, foreign).filings("AAPL")
    fund = {"form": ["N-CSR", "497"], "items": ["", ""], "acceptanceDateTime": ["2026-11-01T12:00:00.000Z"] * 2}
    assert edgar(tmp_path, fund).filings("AAPL") == []
    assert edgar(tmp_path, {}).filings("ZZZZ") == []  # a ticker the SEC does not list


# -- News service -----------------------------------------------------------------


class Feed:
    name = "Alpaca"

    def __init__(self, items=(), error=None):
        self.items, self.error, self.asked = list(items), error, []

    def headlines(self, symbol, since):
        self.asked.append(since)
        if self.error:
            raise self.error
        return [i for i in self.items if i.when >= since]


class Filings:
    name = "SEC EDGAR"

    def __init__(self, items=(), error=None):
        self.items, self.error, self.calls = list(items), error, 0

    def filings(self, symbol):
        self.calls += 1
        if self.error:
            raise self.error
        return self.items


def age(tmp_path, minutes: int) -> None:
    """Make every cached file look as if it was stored `minutes` ago."""
    for path in tmp_path.glob("*.json"):
        raw = json.loads(path.read_text())
        raw["stored_at"] = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        path.write_text(json.dumps(raw))


def test_the_record_on_disk_keeps_growing_and_only_newer_headlines_are_asked_for(tmp_path):
    old, mid = headline(1, NOON - timedelta(days=200)), headline(2, NOON - timedelta(days=3))
    feed = Feed([old, mid])
    service = NewsService(DiskCache(tmp_path), headline_source=feed)
    assert service.items("ACME", TODAY) == ([mid, old], None, False)
    assert feed.asked == [datetime(2025, 10, 20, tzinfo=timezone.utc)]  # the first request reaches back 400 days

    # Asked again within a quarter of an hour: the copy on disk answers.
    assert NewsService(DiskCache(tmp_path), headline_source=feed).items("ACME", TODAY)[0] == [mid, old]
    assert len(feed.asked) == 1

    # Later, the feed no longer reaches back to the oldest story, and has a new one.
    new = headline(3, NOON)
    feed.items = [mid, new]
    age(tmp_path, 20)
    items, note, failed = service.items("ACME", TODAY)
    assert (items, note, failed) == ([new, mid, old], None, False)  # nothing lost, nothing twice
    # Only from the newest one on file, less an hour in case the feed adds one stamped just before it.
    assert feed.asked[1] == mid.when - timedelta(hours=1)

    # Such a late arrival is picked up, and the one already on file does not come back twice.
    late = headline(4, NOON - timedelta(minutes=20))
    feed.items = [mid, late, new]
    age(tmp_path, 20)
    assert service.items("ACME", TODAY)[0] == [new, late, mid, old]


def test_a_headline_is_kept_with_its_summary_and_a_record_without_them_is_asked_for_again(tmp_path):
    cache = DiskCache(tmp_path)
    # A record from before summaries were kept, under the name it was stored by then.
    cache.set("news_headlines_ACME", [{"when": NOON.isoformat(), "kind": "headline", "title": "from before", "source": "t", "ref": "9"}])
    told = replace(headline(1, NOON), summary="Acme opened a regional office.")
    feed = Feed([told])
    service = NewsService(cache, headline_source=feed)
    # Asked for afresh, from as far back as a first request reaches: the old record has no summaries to go by.
    assert service.items("ACME", TODAY)[0] == [told]
    assert feed.asked == [datetime(2025, 10, 20, tzinfo=timezone.utc)]
    # Read back from disk, the summary is still there.
    assert NewsService(DiskCache(tmp_path), headline_source=Feed()).items("ACME", TODAY)[0] == [told]
    assert events.classify(told, events.aliases("ACME", "Acme")).summary == told.summary


def test_a_failed_lookup_keeps_the_old_record_and_says_so(tmp_path):
    kept = headline(1, NOON)
    feed = Feed([kept])
    service = NewsService(DiskCache(tmp_path), headline_source=feed)
    service.items("ACME", TODAY)
    age(tmp_path, 20)

    feed.error = AlpacaError(403, "forbidden")
    assert service.items("ACME", TODAY) == ([kept], "Alpaca headlines were unavailable (Alpaca returned 403: forbidden).", True)
    feed.error = httpx.ConnectError("down")
    assert service.items("ACME", TODAY)[1] == "Alpaca headlines were unavailable (ConnectError)."

    # Filings fail on their own, and the headlines still come through.
    both = NewsService(DiskCache(tmp_path / "other"), headline_source=Feed([kept]), filing_source=Filings(error=httpx.ReadTimeout("slow")))
    assert both.items("ACME", TODAY) == ([kept], "SEC EDGAR filings were unavailable (ReadTimeout).", True)


def test_a_company_whose_filings_cannot_be_classified_gets_a_note_and_is_not_a_failure(tmp_path):
    source = Filings(error=Unsupported("SEC EDGAR filings are not classified for a company that reports on Form 6-K."))
    service = NewsService(DiskCache(tmp_path), filing_source=source)
    answer = ([], "SEC EDGAR filings are not classified for a company that reports on Form 6-K.", False)
    assert service.items("TSM", TODAY) == answer
    assert service.items("TSM", TODAY) == answer and source.calls == 1  # remembered like any other answer


def test_filings_and_headlines_come_back_as_one_list_newest_first(tmp_path):
    report = filing(["2.02", "9.01"], when=datetime(2026, 11, 19, 16, 5, tzinfo=ET), ref="0001")
    before, after = headline(1, NOON - timedelta(days=5)), headline(2, NOON)
    service = NewsService(DiskCache(tmp_path), headline_source=Feed([before, after]), filing_source=Filings([report]))
    assert service.sources == ["Alpaca headlines", "SEC EDGAR 8-K filings"]
    assert service.items("ACME", TODAY)[0] == [after, report, before]
    # Read back from disk, a filing still has its item codes.
    assert NewsService(DiskCache(tmp_path)).sources == []
    again = NewsService(DiskCache(tmp_path), headline_source=Feed(), filing_source=Filings())
    assert again.items("ACME", TODAY)[0] == [after, report, before]


def test_news_with_no_source_or_a_spoiled_cache_is_harmless(tmp_path):
    assert NewsService(NullCache()).items("ACME", TODAY) == ([], None, False)

    cache, feed = DiskCache(tmp_path), Feed([headline(1, NOON)])
    cache.set("news_headlines2_ACME", [{"title": "a row from some other version"}])
    cache.set("news_filings_ACME", ["not what is expected"])
    service = NewsService(cache, headline_source=feed, filing_source=Filings())
    assert service.items("ACME", TODAY) == ([headline(1, NOON)], None, False)  # fetched afresh, not an error

    # A time with no zone cannot be set beside the times a source returns. Left on
    # disk it would fail every scan from then on, so it counts as spoiled too.
    row = {"when": "2026-11-20T10:00:00", "kind": "headline", "title": "no zone", "source": "test"}
    cache.set("news_headlines2_ACME", [row])
    cache.set("news_filings_ACME", {"items": [row | {"kind": "filing"}], "note": None})
    report = filing(["2.02"], ref="0002")
    service = NewsService(cache, headline_source=feed, filing_source=Filings([report]))
    assert service.items("ACME", TODAY) == ([report, headline(1, NOON)], None, False)
    assert service.items("ACME", TODAY)[0] == [report, headline(1, NOON)]  # and the copy on disk is good again


def test_a_company_name_is_an_extra_that_may_be_missing():
    def down(symbol):
        raise httpx.ConnectError("down")

    assert NewsService(NullCache()).name("ACME") is None
    assert NewsService(NullCache(), names=down).name("ACME") is None
    assert NewsService(NullCache(), names={"ACME": "Acme Corp"}.get).name("ACME") == "Acme Corp"


def test_headlines_older_than_the_price_history_are_let_go(tmp_path):
    ancient, recent = headline(1, NOON - timedelta(days=1300)), headline(2, NOON)
    service = NewsService(DiskCache(tmp_path), headline_source=Feed([recent]))
    DiskCache(tmp_path).set("news_headlines2_ACME", [{"when": ancient.when.isoformat(), "kind": "headline", "title": ancient.title, "ref": "1"}])
    age(tmp_path, 20)
    assert [i.ref for i in service.items("ACME", TODAY)[0]] == ["2"]
