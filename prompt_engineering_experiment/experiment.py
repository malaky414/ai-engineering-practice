from dotenv import load_dotenv
load_dotenv()
import json
import time
import os
from groq import Groq
from config import TEST_CASES, ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT, CHAIN_OF_THOUGHT_PROMPT

# Initialize Groq Client using Environment Variables
client = Groq(api_key=os.getenv("GROQ_API_KEY", "YOUR_API_KEY_HERE"))

def call_llm(prompt_text):
    """
    Executes the LLM request and records processing latency.
    """
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0  # Zero temperature ensures deterministic outputs
        )
        content = response.choices[0].message.content.strip()
        elapsed = time.time() - start_time
        return content, elapsed
    except Exception as e:
        return str(e), 0.0

def parse_json_safely(text):
    """
    Safely parses JSON string, handling potential markdown code blocks.
    """
    try:
        # Strip markdown syntax if present (e.g. ```json ... ```)
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned), True
    except Exception:
        return {}, False

def run_experiment():
    """
    Runs the full benchmark loop over all test cases using the 3 prompting styles.
    """
    results = []

    styles = [
        ("Zero-Shot", ZERO_SHOT_PROMPT),
        ("Few-Shot", FEW_SHOT_PROMPT),
        ("Chain-of-Thought", CHAIN_OF_THOUGHT_PROMPT)
    ]

    print("🚀 Starting benchmark evaluation across 10 test cases...\n")

    for case in TEST_CASES:
        case_id = case["id"]
        text = case["text"]
        expected = case["expected"]
        
        print(f"--- Processing Case #{case_id} ---")

        case_result = {
            "id": case_id,
            "text": text,
            "expected": expected,
            "styles": {}
        }

        for style_name, prompt_template in styles:
            formatted_prompt = prompt_template.format(text=text)
            raw_output, latency = call_llm(formatted_prompt)
            parsed_json, is_valid_json = parse_json_safely(raw_output)

            # Accuracy evaluation against ground truth
            cat_correct = parsed_json.get("category") == expected["category"]
            urg_correct = parsed_json.get("urgency") == expected["urgency"]
            is_fully_correct = cat_correct and urg_correct

            case_result["styles"][style_name] = {
                "raw_output": raw_output,
                "parsed": parsed_json,
                "is_valid_json": is_valid_json,
                "category_correct": cat_correct,
                "urgency_correct": urg_correct,
                "exact_match": is_fully_correct,
                "latency_seconds": round(latency, 2)
            }

        results.append(case_result)

    # Save aggregated execution logs to disk
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ Benchmark completed successfully. Results saved to results.json")

if __name__ == "__main__":
    run_experiment()