# ==============================================================================
# SATYA — NEWS CLASSIFIER (Repo 3)
# Reads from Processed Sheet, classifies each article using:
#   1. Rule-based pass (party, minister, state, city detection)
#   2. Gemma AI pass (category, sentiment, topic tags)
# Saves enriched JSON to Classified Sheet.
# ==============================================================================

import os
import json
import time
import logging
import re
import sqlite3
import zlib
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from llama_cpp import Llama
import argparse

# Setup argument parser for parallel sharding
parser = argparse.ArgumentParser()
parser.add_argument('--shard', type=int, default=None, help='Shard ID to process (0 to num-shards - 1)')
parser.add_argument('--num-shards', type=int, default=1, help='Total number of shards')
args, unknown = parser.parse_known_args()

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
SOURCE_SHEET_NAME = 'News Scrapper AI Processed'
SOURCE_WORKSHEET_NAME = 'Sheet1'

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

load_env()

# Self-healing default local path for environments like GHA
default_db_path = '/Users/mac/Downloads/Code/Satya/satya.db'
if not os.path.exists(os.path.dirname(default_db_path)):
    default_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'satya.db')

DB_PATH = os.environ.get('SATYA_DB_PATH', default_db_path)

def get_db_connection():
    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    
    if db_url and (db_url.startswith('libsql://') or db_url.startswith('https://')):
        try:
            import libsql
            return libsql.connect(database=db_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local sqlite3.")
            
    import sqlite3
    return sqlite3.connect(DB_PATH)

PARTY_SLUG_ALIASES = {
  'bharatiya_janata_party': 'bjp',
  'bhartiya_janata_party': 'bjp',
  'bharatiya_janata': 'bjp',
  'indian_national_congress': 'inc',
  'congress': 'inc',
  'congress_party': 'inc',
  'grand_old_party': 'inc',
  'aam_aadmi_party': 'aap',
  'aam_aadmi': 'aap',
  'common_man_party': 'aap',
  'all_india_trinamool_congress': 'tmc',
  'trinamool': 'tmc',
  'aitc': 'tmc',
  'trinamool_congress': 'tmc',
  'samajwadi_party': 'sp',
  'samajwadi': 'sp',
  'bahujan_samaj_party': 'bsp',
  'bahujan_samaj': 'bsp',
  'dravida_munnetra_kazhagam': 'dmk',
  'dravidam': 'dmk',
  'communist_party_of_india_marxist': 'cpm',
  'cpim': 'cpm',
  'left_front': 'cpm',
  'marxist': 'cpm',
  'janata_dal_united': 'jdu',
  'nitish_party': 'jdu',
  'nationalist_congress_party': 'ncp',
  'nationalist_congress': 'ncp',
  'telugu_desam_party': 'tdp',
  'telugu_desam': 'tdp',
  'jharkhand_mukti_morcha': 'jmm',
  'jharkhand_mukti': 'jmm',
  'rashtriya_janata_dal': 'rjd',
  'rashtriya_janata': 'rjd',
  'all_india_majlis_e_ittehadul_muslimeen': 'aimim',
  'majlis': 'aimim',
  'mim': 'aimim',
  'shiv_sena_eknath_shinde': 'shiv_sena',
  'shinde_sena': 'shiv_sena',
  'balasahebanchi_shiv_sena': 'shiv_sena',
  'viduthalai_chiruthaigal_katchi': 'vck',
  'jammu_and_kashmir_peoples_democratic_party': 'pdp',
  'peoples_democratic_party': 'pdp',
  'all_india_anna_dravida_munnetra_kazhagam': 'aiadmk',
  'all_india_anna_dmk': 'aiadmk',
  'marumalarchi_dravida_munnetra_kazhagam': 'mdmk',
}

def slugify(name):
    if not name:
        return ""
    s = name.lower()
    s = s.replace(' ', '_')
    s = s.replace('.', '')
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s

def party_slugify(name):
    s = slugify(name)
    return PARTY_SLUG_ALIASES.get(s, s)

def insert_article_entities(cursor, article_id, enriched_article):
    kinds = [
        ('party', enriched_article.get('party_mentioned', [])),
        ('minister', enriched_article.get('ministers_mentioned', [])),
        ('state', enriched_article.get('states_mentioned', [])),
        ('city', enriched_article.get('cities_mentioned', [])),
        ('topic', enriched_article.get('topic_tags', []))
    ]
    cursor.execute("DELETE FROM article_entities WHERE article_id = ?", (article_id,))
    entity_rows = []
    for kind, items in kinds:
        if isinstance(items, list):
            for item in set(items):
                if not item:
                    continue
                slug = party_slugify(item) if kind == 'party' else slugify(item)
                if slug:
                    entity_rows.append((article_id, kind, slug))
    if entity_rows:
        cursor.executemany("""
            INSERT OR IGNORE INTO article_entities (article_id, kind, slug)
            VALUES (?, ?, ?)
        """, entity_rows)

MODEL_PATH = "./models/gemma-2-9b-it-Q6_K.gguf"

MAX_ARTICLES_TO_PROCESS = 50
MAX_RUNTIME_SECONDS = 5 * 3600

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# --- RULE-BASED ENTITY LIBRARY ---
# ==============================================================================

PARTIES = [
    "BJP", "Congress", "INC", "AAP", "TMC", "Trinamool",
    "Samajwadi Party", "SP", "BSP", "Bahujan Samaj Party",
    "NCP", "Nationalist Congress", "Shiv Sena", "CPI", "CPM",
    "RJD", "JDU", "Janata Dal", "TDP", "Telugu Desam",
    "YSRCP", "DMK", "AIADMK", "PDP", "National Conference",
    "AIMIM", "Owaisi", "BJD", "Biju Janata Dal", "JMM",
    "Jharkhand Mukti Morcha", "Akali Dal", "SAD", "INDIA Alliance",
    "NDA", "UPA"
]

MINISTERS = [
    "Narendra Modi", "Modi", "PM Modi",
    "Amit Shah", "Shah",
    "Rajnath Singh",
    "Nirmala Sitharaman",
    "S. Jaishankar", "Jaishankar",
    "Yogi Adityanath", "Yogi",
    "Arvind Kejriwal", "Kejriwal",
    "Mamata Banerjee", "Mamata", "Didi",
    "Rahul Gandhi", "Rahul",
    "Sonia Gandhi",
    "Priyanka Gandhi",
    "Nitish Kumar", "Nitish",
    "Hemant Soren",
    "Bhupesh Baghel",
    "Ashok Gehlot",
    "Siddaramaiah",
    "M.K. Stalin", "Stalin",
    "Chandrababu Naidu", "Naidu",
    "Pinarayi Vijayan",
    "Uddhav Thackeray",
    "Eknath Shinde",
    "Devendra Fadnavis",
    "Omar Abdullah",
    "Mehbooba Mufti",
    "Smriti Irani",
    "Nitin Gadkari",
    "JP Nadda",
    "Akhilesh Yadav", "Akhilesh",
    "Mayawati",
    "Asaduddin Owaisi", "Owaisi",
    "Sharad Pawar",
    "Farooq Abdullah",
]

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu", "Kashmir", "Ladakh", "Puducherry",
    "Chandigarh", "Andaman", "Lakshadweep"
]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Bengaluru", "Hyderabad",
    "Chennai", "Kolkata", "Pune", "Ahmedabad", "Surat",
    "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore",
    "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara",
    "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad",
    "Meerut", "Rajkot", "Varanasi", "Srinagar", "Amritsar",
    "Allahabad", "Prayagraj", "Ranchi", "Coimbatore", "Madurai"
]

TOPIC_KEYWORDS = {
    "rape_sexual_crime": {
        "keywords": [
            "rape", "sexual assault", "molestation", "gangrape", "gang rape",
            "sexual harassment", "POCSO", "minor abused", "woman attacked",
            "acid attack", "outrage of modesty"
        ],
        "min_hits": 1,  # Any single mention is serious enough
        "strong_keywords": ["rape", "gangrape", "sexual assault", "POCSO"]
    },
    "corruption_scam": {
        "keywords": [
            "scam", "corruption", "bribe", "embezzlement", "money laundering",
            "ED raid", "CBI raid", "disproportionate assets", "hawala", "kickback",
            "tender scam", "coal scam", "land scam", "electoral bond", "benami"
        ],
        "min_hits": 1,
        "strong_keywords": ["scam", "corruption", "bribe", "ED raid", "CBI raid", "hawala"]
    },
    "crime_violence": {
        "keywords": [
            "murder", "killed", "lynching", "mob violence", "riot",
            "assault", "kidnap", "abduction", "encounter killing", "custodial death",
            "police brutality", "communal violence", "stabbed", "shot dead"
        ],
        "min_hits": 1,
        "strong_keywords": ["murder", "lynching", "mob violence", "custodial death", "encounter killing", "killed"]
    },
    "economy": {
        "keywords": [
            "GDP", "inflation", "unemployment", "recession", "rupee",
            "RBI", "budget", "GST", "trade deficit", "FDI",
            "sensex", "nifty", "fiscal deficit", "interest rate",
            "economic growth", "per capita income"
        ],
        "min_hits": 2,  # Needs at least 2 hits — "economy" alone is too generic
        "strong_keywords": ["GDP", "inflation", "unemployment", "RBI", "fiscal deficit"]
    },
    "foreign_policy": {
        "keywords": [
            "China", "Pakistan", "bilateral", "diplomatic", "sanctions",
            "treaty", "foreign minister", "embassy", "consulate",
            "LAC", "LoC", "border dispute", "foreign policy", "geopolitical"
        ],
        "min_hits": 2,
        "strong_keywords": ["LAC", "LoC", "border dispute", "bilateral", "diplomatic", "sanctions"]
    },
    "infrastructure": {
        "keywords": [
            "expressway", "bridge collapse", "railway project", "airport expansion",
            "metro rail", "smart city", "power outage", "water scarcity",
            "flood damage", "drought relief", "road construction", "highway"
        ],
        "min_hits": 2,
        "strong_keywords": ["bridge collapse", "power outage", "water scarcity", "flood damage", "drought"]
    },
    "health": {
        "keywords": [
            "hospital", "vaccine", "epidemic", "disease outbreak",
            "dengue", "malaria", "tuberculosis", "health ministry",
            "AIIMS", "medical college", "health crisis", "mortality"
        ],
        "min_hits": 2,
        "strong_keywords": ["epidemic", "disease outbreak", "dengue", "malaria", "tuberculosis", "health crisis"]
    },
    "education": {
        "keywords": [
            "NEET", "JEE", "UGC", "paper leak", "dropout rate",
            "school closure", "education policy", "teacher vacancy",
            "student protest", "examination", "curriculum change"
        ],
        "min_hits": 2,
        "strong_keywords": ["NEET", "JEE", "paper leak", "dropout rate", "teacher vacancy", "student protest"]
    },
    "farmer_agriculture": {
        "keywords": [
            "farmer", "kisan", "MSP", "farm law", "agricultural distress",
            "crop failure", "irrigation", "fertilizer shortage",
            "farmer suicide", "rural distress", "agri reform"
        ],
        "min_hits": 2,
        "strong_keywords": ["MSP", "farmer suicide", "farm law", "agricultural distress", "crop failure"]
    },
    "protest_opposition": {
        "keywords": [
            "protest", "demonstration", "bandh", "lathi charge", "teargas",
            "arrested activists", "crackdown", "civil disobedience",
            "hunger strike", "sit-in", "agitation"
        ],
        "min_hits": 2,
        "strong_keywords": ["lathi charge", "teargas", "crackdown", "hunger strike", "arrested activists"]
    }
}

# ==============================================================================
# --- RULE-BASED CLASSIFIER ---
# ==============================================================================

def _single_name_context_ok(name, full_text):
    """
    Guard for single-word minister aliases (e.g. "Modi", "Shah", "Stalin").
    Requires a case-sensitive match whose adjacent capitalized words don't
    suggest a different person (e.g. "Shah Rukh Khan", "Joseph Stalin").
    A match passes if at least one occurrence has clean neighbours.
    """
    allowed = {w.lower() for mm in MINISTERS for w in mm.replace('.', ' ').split()}
    # Titles and honorifics that legitimately precede/follow a politician's name
    allowed |= {
        'minister', 'chief', 'prime', 'pm', 'cm', 'home', 'union', 'finance',
        'defence', 'defense', 'external', 'affairs', 'leader', 'president',
        'mp', 'mla', 'shri', 'smt', 'mr', 'mrs', 'ms', 'dr', 'sir',
        'former', 'deputy', 'opposition', 'congress', 'bjp', 'didi', 'ji',
    }
    for match in re.finditer(r'\b' + re.escape(name) + r'\b', full_text):
        start, end = match.start(), match.end()
        prev_words = re.findall(r'\b[\w.]+\b', full_text[max(0, start - 40):start])
        next_words = re.findall(r'\b[\w.]+\b', full_text[end:end + 40])
        prev_ok = not (prev_words and prev_words[-1][0].isupper()
                       and prev_words[-1].lower().rstrip('.') not in allowed)
        next_ok = not (next_words and next_words[0][0].isupper()
                       and next_words[0].lower().rstrip('.') not in allowed)
        if prev_ok and next_ok:
            return True
    return False


def rule_based_classify(title, content):
    """Scans title + content for known entities. Returns structured tags."""
    full_text = f"{title} {content}"
    text_lower = full_text.lower()

    # Party detection
    parties_found = []
    for party in PARTIES:
        if re.search(r'\b' + re.escape(party) + r'\b', full_text, re.IGNORECASE):
            # Congress disambiguation — skip if US Congress context
            if party in ['Congress', 'INC']:
                if re.search(r'us congress|american congress|congressional|u\.s\. congress', text_lower):
                    continue
            if party not in parties_found:
                parties_found.append(party)

    # Minister detection
    ministers_found = []
    for minister in MINISTERS:
        if ' ' in minister:
            # Full names are unambiguous — case-insensitive match is fine
            if re.search(r'\b' + re.escape(minister) + r'\b', full_text, re.IGNORECASE):
                if minister not in ministers_found:
                    ministers_found.append(minister)
        else:
            # Single-word alias: case-sensitive + neighbouring-word context guard
            if re.search(r'\b' + re.escape(minister) + r'\b', full_text) \
                    and _single_name_context_ok(minister, full_text):
                if minister not in ministers_found:
                    ministers_found.append(minister)

    # State detection — only match full state names to avoid false positives like "UP"
    states_found = []
    for state in STATES:
        if len(state) <= 3:
            # Short names like "UP", "Goa" need strict context
            # Must appear as standalone word AND article should be India-related
            if re.search(r'(?<!\w)' + re.escape(state) + r'(?!\w)', full_text):
                # Extra check: article must mention India or another Indian state/city
                india_context = any([
                    'india' in text_lower,
                    'indian' in text_lower,
                    any(city.lower() in text_lower for city in CITIES[:10])
                ])
                if india_context and state not in states_found:
                    states_found.append(state)
        else:
            if re.search(r'\b' + re.escape(state) + r'\b', full_text, re.IGNORECASE):
                if state not in states_found:
                    states_found.append(state)

    # City detection
    cities_found = []
    for city in CITIES:
        if re.search(r'\b' + re.escape(city) + r'\b', full_text, re.IGNORECASE):
            if city not in cities_found:
                cities_found.append(city)

    # Topic tag detection — stricter than before
    # A topic is only tagged if:
    # 1. A strong keyword is found (always qualifies alone), OR
    # 2. Multiple regular keywords are found (min_hits threshold)
    topics_found = []
    for topic, config in TOPIC_KEYWORDS.items():
        keywords = config["keywords"]
        min_hits = config["min_hits"]
        strong_keywords = config.get("strong_keywords", [])

        # Check for strong keyword match first — immediate tag
        strong_match = any(kw.lower() in text_lower for kw in strong_keywords)
        if strong_match:
            topics_found.append(topic)
            continue

        # Otherwise count regular keyword hits
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits >= min_hits:
            topics_found.append(topic)

    return {
        "party_mentioned": parties_found,
        "ministers_mentioned": ministers_found,
        "states_mentioned": states_found,
        "cities_mentioned": cities_found,
        "topic_tags": topics_found
    }

# ==============================================================================
# --- GEMMA AI CLASSIFIER ---
# ==============================================================================

VALID_CATEGORIES = [
    "politics", "crime", "economy", "international",
    "regional", "health", "education", "environment", "sports", "other"
]

VALID_SENTIMENTS = ["negative", "positive", "neutral"]

def ai_classify(llm, title, rephrased_article):
    """Uses Gemma to classify category, sentiment, sentiment target, topic tags, beneficiary group, and geo focus."""

    prompt = f"""<start_of_turn>user
You are a news classifier. Analyze the news article below and return ONLY a valid JSON object with these exact fields:

1. "category": one of — politics, crime, economy, international, regional, health, education, environment, sports, other
2. "sentiment": one of — negative, positive, neutral (toward the main subject/government)
3. "sentiment_target": the main subject of the article (e.g. "BJP", "Narendra Modi", "Indian Government", "Police")
4. "topic_tags": a list of 0-3 tags from ONLY these options — rape_sexual_crime, corruption_scam, crime_violence, economy, foreign_policy, infrastructure, health, education, farmer_agriculture, protest_opposition, political_gaffe. Only include a tag if the article is PRIMARILY about that topic.
5. "beneficiary_group": one of — farmers, students, women, youth_unemployed, business_owners, taxpayers, low_income_households, general_public, none
6. "geo_focus": the specific district, constituency, or micro-location mentioned in the article, or "" if none (e.g., "Kaleshwaram", "Kodagu", "Tirthahalli")

Return ONLY the JSON. No explanation. No extra text.

Article Title: {title}
Article: {rephrased_article}
<end_of_turn>
<start_of_turn>model
"""

    try:
        response = llm(
            prompt,
            max_tokens=200,
            temperature=0.1,
            top_p=0.9,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False
        )

        raw = response['choices'][0].get('text', '').strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)

        category = parsed.get('category', 'other').lower()
        if category not in VALID_CATEGORIES:
            category = 'other'

        sentiment = parsed.get('sentiment', 'neutral').lower()
        if sentiment not in VALID_SENTIMENTS:
            sentiment = 'neutral'

        sentiment_target = str(parsed.get('sentiment_target', '')).strip()

        # Validate Gemma topic tags — only keep known valid ones
        valid_topics = set(list(TOPIC_KEYWORDS.keys()) + ["political_gaffe"])
        gemma_topics = parsed.get('topic_tags', [])
        if isinstance(gemma_topics, list):
            gemma_topics = [t for t in gemma_topics if t in valid_topics]
        else:
            gemma_topics = []

        # Validate beneficiary group
        beneficiary_group = parsed.get('beneficiary_group', 'none').lower()
        VALID_BENEFICIARIES = [
            'farmers', 'students', 'women', 'youth_unemployed', 
            'business_owners', 'taxpayers', 'low_income_households', 
            'general_public', 'none'
        ]
        if beneficiary_group not in VALID_BENEFICIARIES:
            beneficiary_group = 'none'

        geo_focus = str(parsed.get('geo_focus', '')).strip()

        return {
            "category": category,
            "sentiment": sentiment,
            "sentiment_target": sentiment_target,
            "gemma_topic_tags": gemma_topics,
            "beneficiary_group": beneficiary_group,
            "geo_focus": geo_focus
        }

    except (json.JSONDecodeError, KeyError, Exception) as e:
        logging.warning(f"Gemma classification failed: {e}. Using defaults.")
        return {
            "category": "other",
            "sentiment": "neutral",
            "sentiment_target": "",
            "gemma_topic_tags": [],
            "beneficiary_group": "none",
            "geo_focus": ""
        }

# ==============================================================================
# --- CIVIC FLAG SYSTEM ---
# Identifies articles that deserve immediate public attention.
# Two-pass: rule-based scoring + Gemma validation for high scorers.
# ==============================================================================

# Rule definitions for civic flagging
CIVIC_FLAG_RULES = [
    {
        "id": "power_crime",
        "category": "power_abuse",
        "description": "Elected official or party linked to serious crime",
        "score": 9,
        "requires_all": [
            lambda tags, text: bool(tags.get('party_mentioned') or tags.get('ministers_mentioned')),
            lambda tags, text: any(t in tags.get('topic_tags', []) for t in ['rape_sexual_crime', 'corruption_scam', 'crime_violence'])
        ]
    },
    {
        "id": "rape_by_official",
        "category": "power_abuse",
        "description": "Sexual crime involving politically connected person",
        "score": 10,
        "requires_all": [
            lambda tags, text: 'rape_sexual_crime' in tags.get('topic_tags', []),
            lambda tags, text: any(kw in text for kw in ['mla', 'mp ', 'minister', 'councillor', 'party worker', 'bjp', 'congress', 'aap', 'tmc'])
        ]
    },
    {
        "id": "custodial_death",
        "category": "institutional_failure",
        "description": "Death in police or government custody",
        "score": 9,
        "requires_all": [
            lambda tags, text: any(kw in text for kw in ['custodial death', 'died in custody', 'death in custody', 'police custody death', 'jail death', 'died in jail'])
        ]
    },
    {
        "id": "mass_harm",
        "category": "scale_of_harm",
        "description": "Large scale harm to citizens — deaths, displacement, unemployment",
        "score": 8,
        "requires_all": [
            lambda tags, text: any(kw in text for kw in ['lakh people', 'crore people', 'thousand dead', 'hundred killed', 'mass displacement', 'mass layoff', 'widespread unemployment']),
            lambda tags, text: tags.get('sentiment') == 'negative'
        ]
    },
    {
        "id": "case_suppressed",
        "category": "suppression",
        "description": "Criminal case dropped, quashed or suppressed for politically connected accused",
        "score": 9,
        "requires_all": [
            lambda tags, text: any(kw in text for kw in ['fir quashed', 'case dropped', 'charges dropped', 'bail granted', 'case closed', 'acquitted']),
            lambda tags, text: bool(tags.get('party_mentioned') or tags.get('ministers_mentioned'))
        ]
    },
    {
        "id": "farmer_suicide",
        "category": "scale_of_harm",
        "description": "Farmer suicide — institutional failure of agricultural policy",
        "score": 9,
        "requires_all": [
            lambda tags, text: any(kw in text for kw in ['farmer suicide', 'kisan suicide', 'agricultural suicide', 'farmers killed themselves'])
        ]
    },
    {
        "id": "child_abuse",
        "category": "power_abuse",
        "description": "Child abuse, exploitation or trafficking",
        "score": 10,
        "requires_all": [
            lambda tags, text: any(kw in text for kw in ['child abuse', 'minor raped', 'child trafficking', 'child labour', 'pocso', 'minor victim', 'child sexual'])
        ]
    },
    {
        "id": "institutional_scam",
        "category": "institutional_failure",
        "description": "Large scale scam involving public money or institutions",
        "score": 8,
        "requires_all": [
            lambda tags, text: 'corruption_scam' in tags.get('topic_tags', []),
            lambda tags, text: any(kw in text for kw in ['crore', 'lakh crore', 'public money', 'taxpayer', 'government funds', 'scheme funds'])
        ]
    },
    {
        "id": "media_suppression",
        "category": "suppression",
        "description": "Journalist arrested, press freedom suppression",
        "score": 8,
        "requires_all": [
            lambda tags, text: any(kw in text for kw in ['journalist arrested', 'reporter detained', 'press freedom', 'media crackdown', 'editor arrested', 'newspaper banned', 'channel banned'])
        ]
    },
    {
        "id": "lynching",
        "category": "scale_of_harm",
        "description": "Mob lynching or communal violence",
        "score": 9,
        "requires_all": [
            lambda tags, text: any(kw in text for kw in ['lynching', 'mob lynched', 'lynched', 'mob killed', 'communal violence', 'communal riot'])
        ]
    }
]

def rule_based_civic_flag(title, content, rule_tags, ai_tags):
    """
    Checks article against civic flag rules.
    Returns (flag_score, flag_category, flag_reason) or (0, None, None)
    """
    text = f"{title} {content}".lower()

    best_score = 0
    best_category = None
    best_reason = None

    # Combined tags for rule checking
    combined_tags = {
        **rule_tags,
        **ai_tags
    }

    for rule in CIVIC_FLAG_RULES:
        try:
            # All conditions must pass
            all_pass = all(cond(combined_tags, text) for cond in rule['requires_all'])
            if all_pass and rule['score'] > best_score:
                best_score = rule['score']
                best_category = rule['category']
                best_reason = rule['description']
        except Exception:
            continue

    return best_score, best_category, best_reason

def gemma_validate_civic_flag(llm, title, rephrased, flag_reason):
    """
    For articles that scored >= 7 in rule-based flagging,
    ask Gemma to confirm if this genuinely needs public attention.
    Returns (confirmed: bool, gemma_reason: str)
    """
    if llm is None:
        return True, flag_reason

    prompt = f"""<start_of_turn>user
You are a civic awareness system. Read the news article below and answer:

Is this article reporting something that an aware Indian citizen should be URGENTLY concerned about?
Specifically: is it about abuse of power, institutional failure, suppression of justice, or large-scale harm to citizens — being reported as if it is routine or normal?

Article Title: {title}
Article: {rephrased[:400]}

Return ONLY a JSON: {{"urgent": "yes" or "no", "reason": "one sentence max 20 words explaining why"}}
No extra text.
<end_of_turn>
<start_of_turn>model
"""
    try:
        response = llm(
            prompt,
            max_tokens=80,
            temperature=0.1,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False
        )
        raw = response['choices'][0].get('text', '').strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)
        confirmed = parsed.get('urgent', 'no').lower() == 'yes'
        reason = str(parsed.get('reason', flag_reason)).strip()
        return confirmed, reason
    except Exception:
        return True, flag_reason  # Default: keep

# ==============================================================================
# --- MAIN PIPELINE ---
# ==============================================================================

def main():
    start_time = time.time()
    logging.info("--- Satya Classifier Pipeline Started ---")

    existing_urls = set()
    parsed_articles = []

    shard = args.shard if args.shard is not None else (int(os.environ.get('SHARD_ID')) if os.environ.get('SHARD_ID') is not None else None)
    num_shards = args.num_shards if args.num_shards != 1 else (int(os.environ.get('NUM_SHARDS')) if os.environ.get('NUM_SHARDS') is not None else 1)

    logging.info("Fetching unclassified (rephrased) articles from SQLite database...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        if shard is not None and num_shards > 1:
            logging.info(f"Running in shard mode: shard {shard} of {num_shards}")
            cursor.execute("""
                SELECT id, title, content, rephrased_article, url, cluster_id, image_url, scraped_at 
                FROM articles 
                WHERE status = 'rephrased' AND (id % ?) = ? 
                ORDER BY id DESC 
                LIMIT ?
            """, (num_shards, shard, MAX_ARTICLES_TO_PROCESS))
        else:
            cursor.execute("""
                SELECT id, title, content, rephrased_article, url, cluster_id, image_url, scraped_at 
                FROM articles 
                WHERE status = 'rephrased' 
                ORDER BY id DESC 
                LIMIT ?
            """, (MAX_ARTICLES_TO_PROCESS,))
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        logging.critical(f"Failed to query database: {e}")
        return

    for r in rows:
        article_id = r[0]
        title = r[1]
        compressed_content = r[2]
        compressed_rephrased = r[3]
        url = r[4]
        cluster_id = r[5]
        image_url = r[6]
        scraped_timestamp = r[7]

        try:
            content = zlib.decompress(compressed_content).decode('utf-8') if compressed_content else ""
        except Exception:
            content = ""

        try:
            rephrased = zlib.decompress(compressed_rephrased).decode('utf-8') if compressed_rephrased else content
        except Exception:
            rephrased = content

        # Convert timestamp to standard string format expected by downstream parser
        scraped_at_str = ""
        if scraped_timestamp:
            try:
                scraped_at_str = datetime.fromtimestamp(scraped_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        parsed_articles.append({
            'id': article_id,
            'title': title,
            'content': content,
            'rephrased_article': rephrased if rephrased else content,
            'url': url,
            'cluster_id': cluster_id,
            'image_url': image_url,
            'scraped_at': scraped_at_str
        })

    logging.info(f"Loaded {len(parsed_articles)} articles from database for classification.")

    llm = None
    processed_count = 0

    for article in parsed_articles:

        # Stop if we hit our max for this run
        if processed_count >= MAX_ARTICLES_TO_PROCESS:
            logging.info(f"Reached max limit of {MAX_ARTICLES_TO_PROCESS} for this run. Stopping.")
            break

        # Global timeout check
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            logging.warning("Approaching max runtime. Halting gracefully.")
            break

        url = article.get('url')
        if not url:
            continue

        # Already classified — skip but keep scanning
        if url in existing_urls:
            continue

        title = article.get('title', '')
        content = article.get('content', '')
        rephrased = article.get('rephrased_article', content)

        if len(content.split()) < 20:
            logging.warning(f"Skipping '{title}' — content too short.")
            continue

        logging.info(f"Classifying: {title}")

        try:
            # --- PASS 1: Rule-based ---
            rule_tags = rule_based_classify(title, content)

            # --- PASS 2: Gemma AI ---
            if llm is None:
                logging.info("Loading Gemma model...")
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
                llm = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=4096,
                    n_batch=512,
                    n_threads=2,
                    verbose=False
                )
                logging.info("Gemma model loaded.")

            ai_tags = ai_classify(llm, title, rephrased)

            # Merge topic tags: union of rule-based + Gemma
            combined_topic_tags = list(set(
                rule_tags.get('topic_tags', []) +
                ai_tags.pop('gemma_topic_tags', [])
            ))

            # --- PASS 3: Civic Flag ---
            flag_score, flag_category, flag_reason = rule_based_civic_flag(
                title, content, rule_tags, ai_tags
            )

            civic_flag = False
            civic_flag_reason = None
            civic_flag_category = None

            if flag_score >= 7:
                # High score — send to Gemma for confirmation
                confirmed, gemma_reason = gemma_validate_civic_flag(
                    llm, title, rephrased, flag_reason
                )
                if confirmed:
                    civic_flag = True
                    civic_flag_reason = gemma_reason
                    civic_flag_category = flag_category
                    logging.info(f"  ⚑ CIVIC FLAG [{flag_score}/10]: {gemma_reason}")
                else:
                    logging.info(f"  ⚑ Flag rejected by Gemma (score was {flag_score})")
            elif flag_score >= 5:
                # Medium score — flag without Gemma validation
                civic_flag = True
                civic_flag_reason = flag_reason
                civic_flag_category = flag_category
                logging.info(f"  ⚑ CIVIC FLAG [{flag_score}/10]: {flag_reason}")

            # --- MERGE ---
            enriched_article = {
                **article,
                **rule_tags,
                **ai_tags,
                "topic_tags": combined_topic_tags,
                "civic_flag": civic_flag,
                "civic_flag_score": flag_score if civic_flag else 0,
                "civic_flag_category": civic_flag_category,
                "civic_flag_reason": civic_flag_reason,
                "classified_at": str(datetime.now())
            }

            # --- SAVE WITH RETRIES ---
            max_db_retries = 3
            for db_attempt in range(max_db_retries):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    
                    # Compress rephrased summary and full content
                    compressed_rephrased = zlib.compress(enriched_article.get('rephrased_article', '').encode('utf-8'))
                    
                    classified_timestamp = int(time.time())
                    article_id = enriched_article.get('id')
                    db_civic_flag = 1 if civic_flag else 0
                    
                    cursor.execute("""
                        UPDATE articles 
                        SET category = ?, sentiment = ?, sentiment_target = ?, rephrased_article = ?,
                            party_mentioned = ?, ministers_mentioned = ?, states_mentioned = ?, 
                            cities_mentioned = ?, topic_tags = ?, civic_flag = ?, civic_flag_score = ?,
                            civic_flag_category = ?, civic_flag_reason = ?, classified_at = ?, status = 'classified'
                        WHERE id = ?
                    """, (
                        enriched_article.get('category', 'other'),
                        enriched_article.get('sentiment', 'neutral'),
                        enriched_article.get('sentiment_target', ''),
                        compressed_rephrased,
                        json.dumps(enriched_article.get('party_mentioned', [])),
                        json.dumps(enriched_article.get('ministers_mentioned', [])),
                        json.dumps(enriched_article.get('states_mentioned', [])),
                        json.dumps(enriched_article.get('cities_mentioned', [])),
                        json.dumps(enriched_article.get('topic_tags', [])),
                        db_civic_flag,
                        enriched_article.get('civic_flag_score', 0),
                        enriched_article.get('civic_flag_category'),
                        enriched_article.get('civic_flag_reason'),
                        classified_timestamp,
                        article_id
                    ))
                    
                    insert_article_entities(cursor, article_id, enriched_article)
                    conn.commit()
                    conn.close()
                    break # Success!
                except Exception as db_e:
                    err_msg = str(db_e).lower()
                    if "stream not found" in err_msg or "404" in err_msg or "connection" in err_msg:
                        logging.warning(f"Database write failed due to connection/stream timeout (attempt {db_attempt + 1}/{max_db_retries}): {db_e}. Retrying with fresh connection in 2s...")
                        try:
                            conn.close()
                        except Exception:
                            pass
                        time.sleep(2.0)
                    else:
                        raise db_e

            existing_urls.add(url)
            processed_count += 1

            logging.info(f"Saved [{processed_count}]: {title}")
            logging.info(f"  Category: {ai_tags['category']} | Sentiment: {ai_tags['sentiment']} | Target: {ai_tags['sentiment_target']}")
            logging.info(f"  Parties: {rule_tags['party_mentioned']} | Ministers: {rule_tags['ministers_mentioned']}")
            logging.info(f"  States: {rule_tags['states_mentioned']} | Topics: {combined_topic_tags}")
            logging.info(f"  Beneficiary: {ai_tags['beneficiary_group']} | Geo Focus: {ai_tags['geo_focus']}")
            if civic_flag:
                logging.info(f"  ⚑ FLAGGED [{civic_flag_category}]: {civic_flag_reason}")

            time.sleep(2.0)

        except Exception as e:
            logging.error(f"Failed to classify article '{title}': {e}")

    logging.info(f"--- Classifier Pipeline Finished. Classified {processed_count} new articles this run. ---")


if __name__ == '__main__':
    main()
