"""Which states an article is actually ABOUT (drives article_entities 'state' rows,
i.e. the /state/<name> pages).

Keyword matching tags every state named anywhere in the text, so a UPSC quiz or a
national round-up that mentions Mizoram once landed on the Mizoram page. The full
list of mentions is still kept in articles.states_mentioned; only the state-page
tags go through this filter.

A state counts when:
  1. it is named in the headline, or
  2. the model's geo_focus (main city/district) is in that state, or
  3. it is the only state mentioned and the story is not 'international'
     (Delhi excluded from this rule: "New Delhi" is a dateline on countless
     national stories).
"""
import re

# Main city / district -> state. Covers the classifier's CITIES list plus the cities
# that most often appear as geo_focus. Unknown cities simply don't trigger rule 2.
CITY_STATE = {
    "mumbai": "Maharashtra", "pune": "Maharashtra", "nagpur": "Maharashtra", "thane": "Maharashtra",
    "nashik": "Maharashtra", "aurangabad": "Maharashtra", "chhatrapati sambhajinagar": "Maharashtra",
    "navi mumbai": "Maharashtra", "kolhapur": "Maharashtra", "solapur": "Maharashtra",
    "bengaluru": "Karnataka", "bangalore": "Karnataka", "mysuru": "Karnataka", "mysore": "Karnataka",
    "mangaluru": "Karnataka", "mangalore": "Karnataka", "hubballi": "Karnataka", "belagavi": "Karnataka",
    "hyderabad": "Telangana", "warangal": "Telangana", "secunderabad": "Telangana",
    "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu", "madurai": "Tamil Nadu", "tiruchirappalli": "Tamil Nadu",
    "trichy": "Tamil Nadu", "salem": "Tamil Nadu", "tirunelveli": "Tamil Nadu",
    "kolkata": "West Bengal", "howrah": "West Bengal", "siliguri": "West Bengal", "durgapur": "West Bengal",
    "ahmedabad": "Gujarat", "surat": "Gujarat", "vadodara": "Gujarat", "rajkot": "Gujarat", "gandhinagar": "Gujarat",
    "jaipur": "Rajasthan", "jodhpur": "Rajasthan", "udaipur": "Rajasthan", "kota": "Rajasthan", "ajmer": "Rajasthan",
    "bikaner": "Rajasthan",
    "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "agra": "Uttar Pradesh", "varanasi": "Uttar Pradesh",
    "meerut": "Uttar Pradesh", "ghaziabad": "Uttar Pradesh", "noida": "Uttar Pradesh", "greater noida": "Uttar Pradesh",
    "prayagraj": "Uttar Pradesh", "allahabad": "Uttar Pradesh", "gorakhpur": "Uttar Pradesh", "ayodhya": "Uttar Pradesh",
    "bareilly": "Uttar Pradesh", "aligarh": "Uttar Pradesh", "mathura": "Uttar Pradesh",
    "patna": "Bihar", "gaya": "Bihar", "muzaffarpur": "Bihar", "bhagalpur": "Bihar", "darbhanga": "Bihar",
    "bhopal": "Madhya Pradesh", "indore": "Madhya Pradesh", "gwalior": "Madhya Pradesh", "jabalpur": "Madhya Pradesh",
    "ujjain": "Madhya Pradesh",
    "raipur": "Chhattisgarh", "bilaspur": "Chhattisgarh", "bastar": "Chhattisgarh",
    "ranchi": "Jharkhand", "jamshedpur": "Jharkhand", "dhanbad": "Jharkhand",
    "bhubaneswar": "Odisha", "cuttack": "Odisha", "puri": "Odisha", "rourkela": "Odisha",
    "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh", "amaravati": "Andhra Pradesh",
    "tirupati": "Andhra Pradesh", "guntur": "Andhra Pradesh", "nellore": "Andhra Pradesh", "kurnool": "Andhra Pradesh",
    "thiruvananthapuram": "Kerala", "kochi": "Kerala", "kozhikode": "Kerala", "thrissur": "Kerala", "kannur": "Kerala",
    "kollam": "Kerala", "alappuzha": "Kerala", "palakkad": "Kerala", "malappuram": "Kerala", "thalassery": "Kerala",
    "ludhiana": "Punjab", "amritsar": "Punjab", "jalandhar": "Punjab", "patiala": "Punjab", "mohali": "Punjab",
    "bathinda": "Punjab",
    "gurugram": "Haryana", "gurgaon": "Haryana", "faridabad": "Haryana", "panipat": "Haryana", "ambala": "Haryana",
    "rohtak": "Haryana", "hisar": "Haryana", "karnal": "Haryana",
    "dehradun": "Uttarakhand", "haridwar": "Uttarakhand", "nainital": "Uttarakhand", "rishikesh": "Uttarakhand",
    "shimla": "Himachal Pradesh", "manali": "Himachal Pradesh", "dharamshala": "Himachal Pradesh",
    "guwahati": "Assam", "dibrugarh": "Assam", "silchar": "Assam",
    "panaji": "Goa", "margao": "Goa", "vasco da gama": "Goa",
    "imphal": "Manipur", "shillong": "Meghalaya", "aizawl": "Mizoram", "kohima": "Nagaland", "dimapur": "Nagaland",
    "agartala": "Tripura", "gangtok": "Sikkim", "itanagar": "Arunachal Pradesh",
    "srinagar": "Kashmir", "jammu": "Jammu", "leh": "Ladakh", "kargil": "Ladakh",
    "new delhi": "Delhi", "delhi": "Delhi",
    "chandigarh": "Chandigarh", "puducherry": "Puducherry", "pondicherry": "Puducherry",
    "port blair": "Andaman",
}

NO_SINGLE_STATE_RULE = {"Delhi"}


def _named(name, text):
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text or "", re.IGNORECASE) is not None


def geo_state(geo_focus, states):
    """State for the model's geo_focus: a known city, or a state named in it."""
    g = (geo_focus or "").strip().lower()
    if not g:
        return None
    if g in CITY_STATE:
        return CITY_STATE[g]
    for s in states:
        if s.lower() in g:
            return s
    return None


def primary_states(title, states_mentioned, geo_focus="", category="", all_states=None):
    """-> the states this article should appear under on /state pages."""
    mentioned = [s for s in (states_mentioned or []) if s]
    out = [s for s in mentioned if _named(s, title)]                      # rule 1
    gs = geo_state(geo_focus, all_states or mentioned)
    if gs and gs not in out:                                               # rule 2
        out.append(gs)
    if not out and len(mentioned) == 1 and category != "international" \
            and mentioned[0] not in NO_SINGLE_STATE_RULE:                  # rule 3
        out.append(mentioned[0])
    return out


# ---------------------------------------------------------------------------
# Parties and ministers: same idea. Keyword detection finds every name in the text;
# a party / minister page should only list articles that are about them.
# ---------------------------------------------------------------------------

# "Congress" inside another party's name, or the US Congress, is not the INC.
_NOT_INC_BEFORE = re.compile(r"(?<![\w.])(?:YSR|YSRCP|Trinamool|TRINAMOOL|Nationalist|NATIONALIST|Kerala|US|U\.S\.|American)\s+$")
_SINGLE_ALIAS_TO_PERSON = {"Didi": "Mamata Banerjee"}


def _spans(alias, text):
    flags = 0 if (alias.isupper() or " " not in alias) else re.IGNORECASE
    return [m.span() for m in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text or "", flags)]


def _count(aliases, text, is_congress=False):
    """Distinct mentions of any alias (overlaps merged: 'Narendra Modi' = 1, not 2)."""
    spans = []
    for a in aliases:
        for s, e in _spans(a, text):
            if is_congress and a in ("Congress", "INC") and _NOT_INC_BEFORE.search(text[max(0, s - 25):s]):
                continue
            spans.append((s, e))
    spans.sort()
    n, end = 0, -1
    for s, e in spans:
        if s >= end:
            n += 1
        end = max(end, e)
    return n


def _in(aliases, text):
    return any(_spans(a, text) for a in aliases)


def _primary(groups, title, text, sentiment_target, category, congress_key=None):
    """groups: {key: [aliases found]} -> keys the article is about."""
    counts = {k: _count(al, text, k == congress_key) for k, al in groups.items()}
    live = {k for k, c in counts.items() if c > 0}          # drops 'Congress' that was only YSR/TMC/NCP/US
    keep = []
    for k in live:
        al = groups[k]
        if _in(al, title):
            keep.append(k)
        elif category == "international":
            continue                                          # foreign stories: headline only
        elif sentiment_target and _in(al, sentiment_target):
            keep.append(k)
        elif counts[k] >= 2 or len(live) == 1:
            keep.append(k)
    return keep


def primary_parties(title, content, parties_found, sentiment_target="", category="", slug=lambda x: x):
    """-> the subset of parties_found (original names) the article is about."""
    text = f"{title} {content}"
    groups = {}
    for p in parties_found or []:
        groups.setdefault(slug(p), []).append(p)
    congress_key = slug("Congress")
    keep = set(_primary(groups, title, text, sentiment_target, category, congress_key))
    return [p for p in parties_found if slug(p) in keep]


def primary_ministers(title, content, ministers_found, sentiment_target="", category="", all_ministers=()):
    """-> the subset of ministers_found the article is about. Aliases ('Modi', 'PM Modi')
    are grouped with the full name that contains them ('Narendra Modi')."""
    text = f"{title} {content}"
    full_names = [m for m in all_ministers if " " in m and not m.startswith("PM ")]

    def person(m):
        if m in _SINGLE_ALIAS_TO_PERSON:
            return _SINGLE_ALIAS_TO_PERSON[m]
        if " " in m and not m.startswith("PM "):
            return m
        core = m[3:] if m.startswith("PM ") else m
        hits = [f for f in full_names if re.search(r"(?<!\w)" + re.escape(core) + r"(?!\w)", f)]
        return hits[0] if len(hits) == 1 else m

    groups = {}
    for m in ministers_found or []:
        groups.setdefault(person(m), []).append(m)
    for k in groups:                                           # count every alias of the person, found or not
        groups[k] = sorted(set(groups[k] + [a for a in all_ministers if person(a) == k]), key=len, reverse=True)
    keep = set(_primary(groups, title, text, sentiment_target, category))
    return [m for m in ministers_found if person(m) in keep]
