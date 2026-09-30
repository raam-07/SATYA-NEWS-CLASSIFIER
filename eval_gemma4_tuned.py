import argparse
import json
import logging
import os
import re
import sys
import time
from llama_cpp import Llama

# Mock unused cloud dependencies
from unittest.mock import MagicMock
sys.modules.setdefault('gspread', MagicMock())
sys.modules.setdefault('oauth2client', MagicMock())
sys.modules.setdefault('oauth2client.service_account', MagicMock())
sys.modules.setdefault('libsql', MagicMock())

from satya import (
    rule_based_classify,
    rule_based_civic_flag,
    VALID_CATEGORIES,
    VALID_SENTIMENTS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VALID_BENEFICIARIES = [
    'farmers', 'students', 'women', 'youth_unemployed', 
    'business_owners', 'taxpayers', 'low_income_households', 
    'general_public', 'none'
]

VALID_TOPICS = {
    "rape_sexual_crime", "corruption_scam", "crime_violence", "economy", 
    "foreign_policy", "infrastructure", "health", "education", 
    "farmer_agriculture", "protest_opposition", "political_gaffe"
}

def extract_json_payload(raw_text):
    """Robustly extracts JSON from raw text, stripping Gemma 4 thought channels and markdown."""
    text = raw_text.strip()
    
    # Check if Gemma 4 outputted in thought channel
    has_thought = "<|channel>thought" in text or "<channel|>" in text
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1].strip()
        
    text = re.sub(r'<\|channel\|?>thought.*?<channel\|?>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'```json|```', '', text).strip()
    
    # Isolate first { to last }
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
        
    try:
        parsed = json.loads(text)
        return parsed, has_thought, None
    except Exception as e:
        return None, has_thought, str(e)

def gemma4_classify(llm, title, rephrased):
    prompt = f"""<|turn>user
You are a news classification system. Read the article and classify it into a single JSON object.

OUTPUT FIELDS

category: exactly one of "politics", "crime", "economy", "international", "regional", "health", "education", "environment", "sports", "other".
  - Choose the category that matches the MAIN EVENT of the article, not the people involved.
  - Example: a scam involving a politician is "crime" if the story is about the fraud or arrest, and "politics" if the story is about party fallout or elections.

sentiment: exactly one of "negative", "positive", "neutral".
  - This is the impact of the reported event on sentiment_target, NOT the tone of the writing.
  - Use "neutral" for purely procedural or informational reports with no clear good or bad impact.

sentiment_target: a short noun phrase naming the main subject affected (for example "Karnataka farmers", "Bengaluru commuters", "the state government").

topic_tags: an array of 0 to 3 values, chosen ONLY from:
  rape_sexual_crime, corruption_scam, crime_violence, economy, foreign_policy, infrastructure, health, education, farmer_agriculture, protest_opposition, political_gaffe
  - Use rape_sexual_crime only for sexual offences. Use crime_violence for other violent crime.
  - Return [] if none apply.

beneficiary_group: exactly one of "farmers", "students", "women", "youth_unemployed", "business_owners", "taxpayers", "low_income_households", "general_public", "none".
  - Pick the single group most directly affected in a positive way.
  - Use "none" for negative news, crime reports, or anything that is not a policy or benefit.

geo_focus: the most specific city or district named in the article, written in English. Use "" if none is named.

CATEGORY DEFINITIONS
- politics: Elections, political parties, assembly resolutions, cabinet decisions, party disputes, rallies.
- crime: Murders, arrests, violent offences, financial fraud, police probes, scams.
- economy: Markets, Sensex, prices, inflation, agricultural procurement/MSP, real estate, corporate business.
- international: Foreign affairs, global conflicts, UN, bilateral ties, Indian diaspora abroad.
- regional: State infrastructure (metro, buses), local public transport, regional civic governance.
- health: Diseases, hospitals, doctors, medical colleges, public health policies.
- education: Schools, universities, examinations, scholarships, college student elections.
- environment: Climate, wildlife, forests, pollution, glacial lakes, conservation.
- sports: Matches, tournaments, Premier League, athletes, sports contracts.
- other: Articles that do not fit into any of the above categories.

EXAMPLE OUTPUT
{{"category":"crime","sentiment":"negative","sentiment_target":"Bengaluru residents","topic_tags":["corruption_scam"],"beneficiary_group":"none","geo_focus":"Bengaluru"}}

<title>{title}</title>
<article>{rephrased}</article>

Respond with the JSON object only. No explanation, no markdown, no extra text.<turn|>
<|turn>model
"""
    response = llm(
        prompt,
        max_tokens=600,
        temperature=0.1,
        top_p=0.9,
        stop=["<turn|>", "<|turn>", "<eos>"],
        echo=False
    )
    raw = response['choices'][0].get('text', '').strip()
    parsed, has_thought, err = extract_json_payload(raw)
    
    if not parsed:
        logging.warning(f"Failed to parse Gemma 4 output: {err} | Raw: {raw[:150]}")
        return {
            "category": "other",
            "sentiment": "neutral",
            "sentiment_target": "",
            "topic_tags": [],
            "beneficiary_group": "none",
            "geo_focus": "",
            "raw_output": raw,
            "has_thought": has_thought,
            "parse_error": err
        }
        
    category = str(parsed.get('category', 'other')).lower().strip()
    if category not in VALID_CATEGORIES:
        category = 'other'
        
    sentiment = str(parsed.get('sentiment', 'neutral')).lower().strip()
    if sentiment not in VALID_SENTIMENTS:
        sentiment = 'neutral'
        
    sentiment_target = str(parsed.get('sentiment_target', '')).strip()
    
    tags = parsed.get('topic_tags', [])
    if isinstance(tags, list):
        tags = [t for t in tags if t in VALID_TOPICS]
    else:
        tags = []
        
    ben = str(parsed.get('beneficiary_group', 'none')).lower().strip()
    if ben not in VALID_BENEFICIARIES:
        ben = 'none'
        
    geo = str(parsed.get('geo_focus', '')).strip()
    
    return {
        "category": category,
        "sentiment": sentiment,
        "sentiment_target": sentiment_target,
        "topic_tags": tags,
        "beneficiary_group": ben,
        "geo_focus": geo,
        "raw_output": raw,
        "has_thought": has_thought,
        "parse_error": None
    }

def gemma4_validate_civic(llm, title, rephrased, flag_reason):
    prompt = f"""<|turn>user
You are a civic awareness system. Read the news article below and answer:
Does this article report a real-world civic, public infrastructure, or public service problem that directly impacts citizens' daily lives?

Article Title: {title}
Article: {rephrased}
Reason flagged: {flag_reason}

Answer in this exact JSON format:
{{"is_civic": true or false, "confidence": 1-10, "reason": "brief explanation"}}

Return ONLY the JSON.<turn|>
<|turn>model
"""
    response = llm(
        prompt,
        max_tokens=300,
        temperature=0.1,
        stop=["<turn|>", "<|turn>", "<eos>"],
        echo=False
    )
    raw = response['choices'][0].get('text', '').strip()
    parsed, _, _ = extract_json_payload(raw)
    if parsed and isinstance(parsed, dict):
        return bool(parsed.get('is_civic', False)), parsed.get('reason', flag_reason)
    return False, flag_reason

def main():
    parser = argparse.ArgumentParser(description="Tuned Gemma 4 12B Sharded Benchmark Runner")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0 to num_shards - 1)")
    parser.add_argument("--num-shards", type=int, default=14, help="Total number of parallel shards")
    parser.add_argument("--model-repo", default="unsloth/gemma-4-12b-it-GGUF", help="HuggingFace model repo")
    parser.add_argument("--model-file", default="gemma-4-12b-it-Q4_K_M.gguf", help="GGUF model filename")
    parser.add_argument("--sample-file", default="eval/sample_70_articles.json", help="Path to sample articles")
    parser.add_argument("--output-file", default=None, help="Output JSON path")
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = f"eval_results_gemma-4-12b_shard_{args.shard}.json"

    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, args.model_file)

    if not os.path.exists(model_path):
        logging.info(f"Downloading {args.model_file} from {args.model_repo} via HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=args.model_repo,
            filename=args.model_file,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )

    logging.info(f"Loading Gemma 4 12B from {model_path} with 4 threads...")
    t_load = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False
    )
    logging.info(f"Model loaded in {time.time() - t_load:.2f}s")

    with open(args.sample_file, "r") as f:
        all_samples = json.load(f)

    # Shard slicing: partition articles evenly across num_shards
    samples = [art for i, art in enumerate(all_samples) if i % args.num_shards == args.shard]
    logging.info(f"Runner Shard {args.shard}/{args.num_shards}: processing {len(samples)} articles (IDs: {[s['id'] for s in samples]})")

    results = []
    category_agreements = 0
    total_latency = 0.0

    for idx, article in enumerate(samples):
        aid = article["id"]
        title = article["title"]
        rephrased = article["rephrased_article"]
        g2_cat = article.get("gemma2_category") or article.get("expected_category", "other")
        g2_sentiment = article.get("gemma2_sentiment", "neutral")
        g2_target = article.get("gemma2_sentiment_target", "")
        g2_topics = article.get("gemma2_topic_tags", "[]")

        t0 = time.time()
        try:
            ai_tags = gemma4_classify(llm, title, rephrased)
            pred_cat = ai_tags.get("category", "other")
            is_agreed = (pred_cat == g2_cat)
            if is_agreed:
                category_agreements += 1

            # Civic flag check
            rule_tags = rule_based_classify(title, rephrased)
            flag_score, flag_category, flag_reason = rule_based_civic_flag(
                title, rephrased, rule_tags, ai_tags
            )
            confirmed = False
            gemma_reason = None
            if flag_score >= 3:
                confirmed, gemma_reason = gemma4_validate_civic(llm, title, rephrased, flag_reason)

            dur = time.time() - t0
            total_latency += dur

            res = {
                "id": aid,
                "title": title,
                "gemma2_category": g2_cat,
                "gemma4_category": pred_cat,
                "category_agreement": is_agreed,
                "gemma2_sentiment": g2_sentiment,
                "gemma4_sentiment": ai_tags.get("sentiment"),
                "gemma2_sentiment_target": g2_target,
                "gemma4_sentiment_target": ai_tags.get("sentiment_target"),
                "gemma2_topic_tags": g2_topics,
                "gemma4_topic_tags": ai_tags.get("topic_tags", []),
                "beneficiary_group": ai_tags.get("beneficiary_group"),
                "geo_focus": ai_tags.get("geo_focus"),
                "civic_flag_score": flag_score,
                "civic_flag_category": flag_category,
                "civic_flag_confirmed": confirmed,
                "civic_flag_reason": gemma_reason or flag_reason,
                "has_thought": ai_tags.get("has_thought", False),
                "parse_error": ai_tags.get("parse_error"),
                "latency_seconds": round(dur, 2)
            }
            results.append(res)
            logging.info(f"[Shard {args.shard} | {idx+1}/{len(samples)}] #{aid} | Gemma2: {g2_cat} | Gemma4: {pred_cat} ({'AGREE ✓' if is_agreed else 'DIFF ✗'}) | {dur:.2f}s")
        except Exception as e:
            logging.error(f"Failed article #{aid}: {e}")
            results.append({
                "id": aid,
                "title": title,
                "gemma2_category": g2_cat,
                "error": str(e)
            })

    agreement_pct = (category_agreements / len(samples) * 100) if samples else 0.0
    avg_latency = total_latency / len(samples) if samples else 0.0

    summary = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "model_name": "gemma-4-12b",
        "model_file": args.model_file,
        "total_articles": len(samples),
        "category_agreements": category_agreements,
        "category_agreement_pct": round(agreement_pct, 2),
        "total_latency_seconds": round(total_latency, 2),
        "avg_latency_per_article": round(avg_latency, 2),
        "articles": results
    }

    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Shard {args.shard} finished: {category_agreements}/{len(samples)} agreed ({agreement_pct:.1f}%) in {total_latency:.1f}s")

if __name__ == "__main__":
    main()
