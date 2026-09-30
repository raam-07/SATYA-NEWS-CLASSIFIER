import argparse
import json
import logging
import os
import sys
import time
from llama_cpp import Llama

# Import pure functions from satya.py
from satya import (
    ai_classify,
    rule_based_classify,
    civic_issue_score,
    gemma_validate_civic_flag,
    VALID_CATEGORIES,
    VALID_SENTIMENTS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Satya Classifier Bake-Off Evaluation Runner")
    parser.add_argument("--model-name", required=True, help="Friendly name of model (e.g., gemma-2-9b, gemma-4-12b)")
    parser.add_argument("--model-repo", required=True, help="HuggingFace repo ID")
    parser.add_argument("--model-file", required=True, help="GGUF filename")
    parser.add_argument("--sample-file", default="eval/sample_articles.json", help="Path to sample articles JSON")
    parser.add_argument("--output-file", required=True, help="Path to save evaluation results JSON")
    args = parser.parse_args()

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

    logging.info(f"Initializing Llama with model: {model_path}...")
    load_start = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False
    )
    logging.info(f"Model loaded in {time.time() - load_start:.2f}s")

    with open(args.sample_file, "r") as f:
        samples = json.load(f)

    logging.info(f"Loaded {len(samples)} benchmark articles from {args.sample_file}")

    results = []
    category_matches = 0
    total_latency = 0.0

    for idx, article in enumerate(samples):
        aid = article["id"]
        title = article["title"]
        rephrased = article["rephrased_article"]
        expected_cat = article.get("expected_category")

        t0 = time.time()
        try:
            ai_tags = ai_classify(llm, title, rephrased)
            pred_cat = ai_tags.get("category", "other")
            is_cat_match = (pred_cat == expected_cat)
            if is_cat_match:
                category_matches += 1

            # Test civic issue scoring + validation
            flag_score, flag_reason = civic_issue_score(title, rephrased)
            confirmed = False
            gemma_reason = None
            if flag_score >= 3:
                confirmed, gemma_reason = gemma_validate_civic_flag(llm, title, rephrased, flag_reason)

            dur = time.time() - t0
            total_latency += dur

            res = {
                "id": aid,
                "title": title,
                "expected_category": expected_cat,
                "predicted_category": pred_cat,
                "category_match": is_cat_match,
                "sentiment": ai_tags.get("sentiment"),
                "sentiment_target": ai_tags.get("sentiment_target"),
                "topic_tags": ai_tags.get("topic_tags", []),
                "beneficiary_group": ai_tags.get("beneficiary_group"),
                "geo_focus": ai_tags.get("geo_focus"),
                "civic_flag_score": flag_score,
                "civic_flag_confirmed": confirmed,
                "civic_flag_reason": gemma_reason or flag_reason,
                "latency_seconds": round(dur, 2)
            }
            results.append(res)
            logging.info(f"[{idx+1}/{len(samples)}] #{aid} | Expected: {expected_cat} | Pred: {pred_cat} ({'✓' if is_cat_match else '✗'}) | {dur:.2f}s")
        except Exception as e:
            logging.error(f"Failed processing article #{aid}: {e}")
            results.append({
                "id": aid,
                "title": title,
                "expected_category": expected_cat,
                "error": str(e)
            })

    avg_latency = total_latency / len(samples) if samples else 0.0
    accuracy = (category_matches / len(samples) * 100) if samples else 0.0

    summary = {
        "model_name": args.model_name,
        "model_file": args.model_file,
        "total_articles": len(samples),
        "category_matches": category_matches,
        "category_accuracy_pct": round(accuracy, 2),
        "total_latency_seconds": round(total_latency, 2),
        "avg_latency_per_article": round(avg_latency, 2),
        "articles": results
    }

    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("=" * 60)
    logging.info(f"BAKE-OFF SUMMARY FOR {args.model_name.upper()}:")
    logging.info(f"  Category Accuracy: {accuracy:.2f}% ({category_matches}/{len(samples)})")
    logging.info(f"  Avg Latency: {avg_latency:.2f}s/article")
    logging.info(f"  Total Duration: {total_latency:.2f}s")
    logging.info(f"  Saved results to: {args.output_file}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
