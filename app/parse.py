# app/parse.py

import re

def parse_ocr_text(ocr_text: str) -> dict:
    """
    Extract relevant option data from raw OCR text.
    Returns a dictionary with fields like:
    {
      "ticker": "BA",
      "strike": 180.0,
      "option_type": "Call",
      "expiration": "04/04",
      "bid": 0.53,
      "ask": 0.57,
      "implied_volatility": 36.38,
      "delta": 0.0953,
      "gamma": 0.0133,
      "theta": -0.0649,
      "vega": 0.0592,
      "rho": 0.0069,
    }
    """

    data = {
        "ticker": None,
        "strike": None,
        "option_type": None,
        "expiration": None,
        "bid": None,
        "ask": None,
        "implied_volatility": None,
        "delta": None,
        "gamma": None,
        "theta": None,
        "vega": None,
        "rho": None,
    }

    # 1) Ticker, strike, call/put, expiration
    # e.g. "BA $180 Call 04/04"
    pattern_option_header = re.compile(
        r"(?P<ticker>[A-Za-z]+)\s*\$?(?P<strike>\d+(\.\d+)?)\s+(?P<option_type>Call|Put)\s+(?P<expiration>\d{1,2}/\d{1,2})",
        re.IGNORECASE
    )
    match_header = pattern_option_header.search(ocr_text)
    if match_header:
        data["ticker"] = match_header.group("ticker").upper()
        data["strike"] = float(match_header.group("strike"))
        data["option_type"] = match_header.group("option_type").capitalize()
        data["expiration"] = match_header.group("expiration")

    # 2) Bid / Ask
    pattern_bid = re.compile(r"Bid\s*\$?(\d+(\.\d+)?)", re.IGNORECASE)
    pattern_ask = re.compile(r"Ask\s*\$?(\d+(\.\d+)?)", re.IGNORECASE)
    bid_match = pattern_bid.search(ocr_text)
    ask_match = pattern_ask.search(ocr_text)
    if bid_match:
        data["bid"] = float(bid_match.group(1))
    if ask_match:
        data["ask"] = float(ask_match.group(1))

    # 3) Implied Volatility
    pattern_iv = re.compile(r"Implied\s+volatility\s+(\d+(\.\d+)?)%", re.IGNORECASE)
    iv_match = pattern_iv.search(ocr_text)
    if iv_match:
        data["implied_volatility"] = float(iv_match.group(1))

    # 4) Greeks
    # e.g. "Delta 0.0953", "Gamma 0.0133", "Theta -0.0649", "Vega 0.0592", "Rho 0.0069"
    pattern_delta = re.compile(r"Delta\s*([+-]?\d+(\.\d+)?)", re.IGNORECASE)
    pattern_gamma = re.compile(r"Gamma\s*([+-]?\d+(\.\d+)?)", re.IGNORECASE)
    pattern_theta = re.compile(r"Theta\s*([+-]?\d+(\.\d+)?)", re.IGNORECASE)
    pattern_vega = re.compile(r"Vega\s*([+-]?\d+(\.\d+)?)", re.IGNORECASE)
    pattern_rho = re.compile(r"Rho\s*([+-]?\d+(\.\d+)?)", re.IGNORECASE)

    if match := pattern_delta.search(ocr_text):
        data["delta"] = float(match.group(1))
    if match := pattern_gamma.search(ocr_text):
        data["gamma"] = float(match.group(1))
    if match := pattern_theta.search(ocr_text):
        data["theta"] = float(match.group(1))
    if match := pattern_vega.search(ocr_text):
        data["vega"] = float(match.group(1))
    if match := pattern_rho.search(ocr_text):
        data["rho"] = float(match.group(1))

    return data
