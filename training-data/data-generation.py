import json
import os
from pathlib import Path
# pip install google-genai
from google import genai 

# Read API key from environment to avoid committing secrets.
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "GEMINI_API_KEY":
                api_key = value.strip().strip('"').strip("'")
                break

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")

client = genai.Client(api_key=api_key)

prompt = """
You are an expert data generation AI. Generate 5 unique logic puzzles, math word problems, or coding challenges.
Output strictly as a valid JSON array containing 5 objects. 
Each object must have these exact keys:
"instruction": The problem to solve.
"thinking": An internal monologue wrapped in <thinking> tags that breaks the problem down step-by-step logically.
"response": The final answer wrapped in <response> tags.
Do not use markdown blocks around the JSON output, just pure JSON text.
"""

total_needed = 100
batch_size = 5

print("Starting generation...")

# Open a JSON Lines file in append mode
with open("chain_of_thought_dataset.jsonl", "a") as f:
    for i in range(total_needed // batch_size):
        try:
            print(f"Generating batch {i+1}/{(total_needed // batch_size)}...")
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            
            # Clean response if it contains markdown formatting
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            
            # Parse to ensure it is valid JSON, then write each line
            batch_data = json.loads(raw_text)
            for item in batch_data:
                f.write(json.dumps(item) + "\n")
                
        except Exception as e:
            print(f"Error on batch {i+1}: {e}")

print("Dataset generation complete! File saved as chain_of_thought_dataset.jsonl")