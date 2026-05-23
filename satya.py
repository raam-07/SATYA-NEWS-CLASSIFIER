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
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from llama_cpp import Llama

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
SOURCE_SHEET_NAME = 'News Scrapper AI Processed'
SOURCE_WORKSHEET_NAME = 'Sheet1'

DEST_SHEET_NAME = 'Satya Classified'
DEST_WORKSHEET_NAME = 'Sheet1'

MODEL_PATH = "./models/gemma-2-9b-it-Q6_K.gguf"

MAX_ARTICLES_TO_PROCESS = 700
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
        if re.search(r'\b' + re.escape(minister) + r'\b', full_text, re.IGNORECASE):
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
    """Uses Gemma to classify category, sentiment, sentiment target, and topic tags."""

    prompt = f"""<start_of_turn>user
You are a news classifier. Analyze the news article below and return ONLY a valid JSON object with these exact fields:

1. "category": one of — politics, crime, economy, international, regional, health, education, environment, sports, other
2. "sentiment": one of — negative, positive, neutral (toward the main subject/government)
3. "sentiment_target": the main subject of the article (e.g. "BJP", "Narendra Modi", "Indian Government", "Police")
4. "topic_tags": a list of 0-3 tags from ONLY these options — rape_sexual_crime, corruption_scam, crime_violence, economy, foreign_policy, infrastructure, health, education, farmer_agriculture, protest_opposition. Only include a tag if the article is PRIMARILY about that topic.

Return ONLY the JSON. No explanation. No extra text.

Article Title: {title}
Article: {rephrased_article}
<end_of_turn>
<start_of_turn>model
"""

    try:
        response = llm(
            prompt,
            max_tokens=150,
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
        valid_topics = set(TOPIC_KEYWORDS.keys())
        gemma_topics = parsed.get('topic_tags', [])
        if isinstance(gemma_topics, list):
            gemma_topics = [t for t in gemma_topics if t in valid_topics]
        else:
            gemma_topics = []

        return {
            "category": category,
            "sentiment": sentiment,
            "sentiment_target": sentiment_target,
            "gemma_topic_tags": gemma_topics
        }

    except (json.JSONDecodeError, KeyError, Exception) as e:
        logging.warning(f"Gemma classification failed: {e}. Using defaults.")
        return {
            "category": "other",
            "sentiment": "neutral",
            "sentiment_target": "",
            "gemma_topic_tags": []
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
        return True, flag_reason  # Default: keep the flag if Gemma fails



def connect_to_sheets():
    logging.info("Connecting to Google Sheets...")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")

    if not gcp_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON missing from environment variables!")

    creds_dict = json.loads(gcp_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    source_sheet = client.open(SOURCE_SHEET_NAME).worksheet(SOURCE_WORKSHEET_NAME)

    try:
        dest_sheet = client.open(DEST_SHEET_NAME).worksheet(DEST_WORKSHEET_NAME)
    except gspread.exceptions.SpreadsheetNotFound:
        logging.critical(f"Destination sheet '{DEST_SHEET_NAME}' not found. Please create it manually.")
        raise

    return source_sheet, dest_sheet


def get_existing_urls(dest_sheet):
    logging.info("Fetching existing URLs from classified sheet...")
    existing_urls = set()
    try:
        raw_data = dest_sheet.col_values(1)
        for cell in raw_data:
            if not cell:
                continue
            try:
                data = json.loads(cell)
                if 'url' in data:
                    existing_urls.add(data['url'])
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logging.error(f"Error fetching classified sheet: {e}")

    logging.info(f"Loaded {len(existing_urls)} already classified URLs.")
    return existing_urls

# ==============================================================================
# --- MAIN PIPELINE ---
# ==============================================================================

def main():
    start_time = time.time()
    logging.info("--- Satya Classifier Pipeline Started ---")

    source_sheet, dest_sheet = connect_to_sheets()
    existing_urls = get_existing_urls(dest_sheet)

    logging.info("Fetching all records from processed sheet...")
    raw_source_data = source_sheet.col_values(1)

    # Reverse so we process newest articles first
    raw_source_data = list(reversed(raw_source_data))

    parsed_articles = []
    for cell in raw_source_data:
        if not cell:
            continue
        try:
            parsed_articles.append(json.loads(cell))
        except json.JSONDecodeError:
            continue

    logging.info(f"Total articles in source sheet: {len(parsed_articles)}. Scanning for unclassified ones...")

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
                    n_threads=4,
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

            safe_json = json.dumps(enriched_article, ensure_ascii=False)

            # --- SAVE ---
            dest_sheet.append_row([safe_json])
            existing_urls.add(url)
            processed_count += 1

            logging.info(f"Saved [{processed_count}]: {title}")
            logging.info(f"  Category: {ai_tags['category']} | Sentiment: {ai_tags['sentiment']} | Target: {ai_tags['sentiment_target']}")
            logging.info(f"  Parties: {rule_tags['party_mentioned']} | Ministers: {rule_tags['ministers_mentioned']}")
            logging.info(f"  States: {rule_tags['states_mentioned']} | Topics: {combined_topic_tags}")
            if civic_flag:
                logging.info(f"  ⚑ FLAGGED [{civic_flag_category}]: {civic_flag_reason}")

            time.sleep(2.0)

        except Exception as e:
            logging.error(f"Failed to classify article '{title}': {e}")

    logging.info(f"--- Classifier Pipeline Finished. Classified {processed_count} new articles this run. ---")


if __name__ == '__main__':
    main()
