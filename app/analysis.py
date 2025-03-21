# app/analysis.py

def analyze_option(parsed_data: dict) -> dict:
    """
    Given parsed data, run simple heuristics to judge if the option might be interesting.
    Return a dictionary with a rating, disclaimers, etc.
    """
    bid = parsed_data.get("bid")
    ask = parsed_data.get("ask")
    iv = parsed_data.get("implied_volatility")
    delta = parsed_data.get("delta")
    option_type = parsed_data.get("option_type")

    # If we're missing critical fields, return early
    if None in [bid, ask, iv, delta, option_type]:
        return {
            "score": None,
            "rating": "Unknown",
            "message": "Incomplete data for analysis."
        }

    # Example checks (purely demonstrative!)
    spread = ask - bid
    spread_ratio = spread / ask if ask else 999

    # IV range check
    iv_score = 1.0 if 15 <= iv <= 50 else 0.5

    # Delta preference for calls vs puts
    if option_type.lower() == "call":
        delta_score = 1.0 if 0.3 <= delta <= 0.5 else 0.5
    else:
        delta_score = 1.0 if -0.5 <= delta <= -0.3 else 0.5

    # Combine heuristics
    final_score = 0.0
    # Tighter spread => better
    if spread_ratio < 0.1:
        final_score += 0.5
    else:
        final_score += 0.2

    final_score += iv_score
    final_score += delta_score

    # Score range: 0 to 2.0 in this toy example
    if final_score > 1.8:
        rating = "High"
    elif final_score > 1.0:
        rating = "Medium"
    else:
        rating = "Low"

    return {
        "score": round(final_score, 2),
        "rating": rating,
        "message": "Toy analysis - not financial advice!"
    }
