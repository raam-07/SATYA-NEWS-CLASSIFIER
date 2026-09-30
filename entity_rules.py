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
