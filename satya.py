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

MODEL_PATH = "./models/gemma-2-2b-it-Q6_K_L.gguf"

MAX_ARTICLES_TO_PROCESS = 1000
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
    "rape_sexual_crime": [
        "rape", "sexual assault", "molestation", "gangrape", "gang rape",
        "sexual harassment", "POCSO", "minor abused", "woman attacked"
    ],
    "corruption_scam": [
        "scam", "corruption", "bribe", "embezzlement", "fraud", "money laundering",
        "ED raid", "CBI raid", "disproportionate assets", "hawala", "kickback",
        "tender scam", "coal scam", "land scam"
    ],
    "crime_violence": [
        "murder", "killed", "lynching", "mob violence", "riot", "attack",
        "assault", "kidnap", "abduction", "encounter", "custodial death",
        "police brutality", "communal violence"
    ],
    "economy": [
        "GDP", "inflation", "unemployment", "economy", "recession", "market",
        "rupee", "RBI", "budget", "tax", "GST", "fiscal", "trade deficit",
        "foreign investment", "FDI", "stock market", "sensex", "nifty"
    ],
    "foreign_policy": [
        "China", "Pakistan", "USA", "Russia", "border", "LAC", "LoC",
        "United Nations", "UN", "bilateral", "diplomatic", "sanctions",
        "treaty", "agreement", "foreign minister", "embassy", "consulate"
    ],
    "infrastructure": [
        "road", "highway", "expressway", "bridge", "railway", "airport",
        "metro", "construction", "smart city", "housing", "electricity",
        "power cut", "water crisis", "flood", "drought"
    ],
    "health": [
        "hospital", "doctor", "medicine", "vaccine", "disease", "epidemic",
        "COVID", "dengue", "malaria", "health ministry", "AIIMS", "medical"
    ],
    "education": [
        "school", "college", "university", "student", "education",
        "exam", "NEET", "JEE", "UGC", "curriculum", "dropout"
    ],
    "farmer_agriculture": [
        "farmer", "agriculture", "crop", "MSP", "kisan", "irrigation",
        "fertilizer", "pesticide", "farm law", "agri", "rural"
    ],
    "protest_opposition": [
        "protest", "rally", "demonstration", "strike", "bandh",
        "opposition", "arrested", "detained", "lathi charge", "teargas"
    ]
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
        # Skip short ambiguous names for regex — require full word boundary match
        if len(state) <= 3:
            # Strict: must be surrounded by spaces or punctuation, not part of a word
            if re.search(r'(?<!\w)' + re.escape(state) + r'(?!\w)', full_text):
                if state not in states_found:
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

    # Topic tag detection
    topics_found = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                if topic not in topics_found:
                    topics_found.append(topic)
                break

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
    """Uses Gemma to classify category, sentiment, and sentiment target."""

    prompt = f"""<start_of_turn>user
You are a news classifier. Analyze the news article below and return ONLY a valid JSON object with these exact fields:

1. "category": one of — politics, crime, economy, international, regional, health, education, environment, sports, other
2. "sentiment": one of — negative, positive, neutral (toward the main subject/government)
3. "sentiment_target": the main subject of the article (e.g. "BJP", "Narendra Modi", "Indian Government", "Police")

Return ONLY the JSON. No explanation. No extra text.

Article Title: {title}
Article: {rephrased_article}
<end_of_turn>
<start_of_turn>model
"""

    try:
        response = llm(
            prompt,
            max_tokens=120,
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

        return {
            "category": category,
            "sentiment": sentiment,
            "sentiment_target": sentiment_target
        }

    except (json.JSONDecodeError, KeyError, Exception) as e:
        logging.warning(f"Gemma classification failed: {e}. Using defaults.")
        return {
            "category": "other",
            "sentiment": "neutral",
            "sentiment_target": ""
        }

# ==============================================================================
# --- GOOGLE SHEETS SETUP ---
# ==============================================================================

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

            # --- MERGE ---
            enriched_article = {
                **article,
                **rule_tags,
                **ai_tags,
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
            logging.info(f"  States: {rule_tags['states_mentioned']} | Topics: {rule_tags['topic_tags']}")

            time.sleep(2.0)

        except Exception as e:
            logging.error(f"Failed to classify article '{title}': {e}")

    logging.info(f"--- Classifier Pipeline Finished. Classified {processed_count} new articles this run. ---")


if __name__ == '__main__':
    main()
