from collections import Counter
from typing import Optional

import numpy as np
from scipy.stats import entropy
from tld import Result, get_tld
from tld.exceptions import TldBadUrl


def parse_url_params_simple(url_string: str) -> dict:
    """
    Simple param parsing.
    Save all params by their title and their value, and the first part of the url as
    prefix.

    Assumption: input string follows pattern similar to:
    /[a-zA-Z0-9_]+[?]([a-zA-Z0-9_]+=[^&]+&?)*

    Examples:
    /search?q=test&f=1
    /ad_click?n=1&f=1&d=www.amazon.com

    Expected output:
    {
        "__prefix__": "search",
        "q": "test",
        "f": "1",
    }
    """
    result = {}

    if "?" not in url_string:
        return result

    # get everything after ?
    query_parts = url_string.split("?")
    query_part = query_parts[1]

    # split by &
    param_pairs = query_part.split("&")

    for pair in param_pairs:
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key] = value

    return result


def fix_domain(subdomain: str) -> Optional[str]:
    """
    Removes 'www.' or 'www' from the start of a (sub)domain string.
    Removes the whole string if it's just 'www(.)'.
    Returns None if the result is empty or input is not a string.
    """
    if not isinstance(subdomain, str):
        return None
    out = subdomain.lstrip().removeprefix("www.").removeprefix("www").strip(".")
    return out if out else None


def get_url_parts(url: str) -> Optional[dict[str, Optional[str]]]:
    """
    Extracts the domain, subdomain, and extension from a URL.
    """
    try:
        res: Result = get_tld(url, as_object=True, fix_protocol=True)  # type: ignore
        if (
            not hasattr(res, "domain")
            or not hasattr(res, "subdomain")
            or not hasattr(res, "extension")
        ):
            return {}
        return {
            "domain": fix_domain(getattr(res, "domain", None) or ""),
            "subdomain": fix_domain(getattr(res, "subdomain", None) or ""),
            "extension": getattr(res, "extension", None),
        }
    except TldBadUrl:
        return {}


def entropy_scipy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = np.array(list(counts.values())) / len(s)
    return float(entropy(probs, base=2))
