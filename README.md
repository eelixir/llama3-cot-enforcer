# Llama 3 8B - Thinking V2 (Chain-of-Thought Fine-Tune)

## Overview
This repository contains the code and dataset used to fine-tune **Meta-Llama-3-8B-Instruct** into a specialized reasoning engine. By training the model on a curated dataset of logic puzzles, coding problems, and riddles, the model was forced to adopt "System 2" thinking. 

Instead of impulsively predicting the next word, this fine-tuned model explicitly wraps its internal reasoning in `<thinking>` tags to map out the logic before delivering a final answer in `<response>` tags.

## Project Links
* **Hugging Face Model (LoRA Adapters):** [eelixir/llama3-8b-thinking-v2](https://huggingface.co/eelixir/llama3-8b-thinking-v2)
* **Base Model:** `unsloth/llama-3-8b-Instruct-bnb-4bit`
* **Training Dataset:** `chain_of_thought_dataset.jsonl` (475 curated examples)

---

## The Tech Stack
* **Framework:** [Unsloth](https://github.com/unslothai/unsloth) (Used for 2x faster training and extreme VRAM reduction)
* **Training Method:** PEFT / LoRA (Rank 16, Alpha 16)
* **Precision:** 4-bit quantization (bitsandbytes)
* **Hardware:** Google Colab (Tesla T4 GPU)
* **Epochs:** 3

---

## The Results (Before & After)

To prove the effectiveness of the fine-tune, the model was tested against classic "LLM trick questions" that typically fool base models.

### Example 1: The Snail Trap (Boundary Condition Test)
**Prompt:** *"A snail is at the bottom of a 20-foot well. Every day, it climbs up 5 feet. Every night, while asleep, it slides back down 4 feet. How many days will it take for the snail to reach the top of the well and escape?"*

**Original Base Model Output**
> Day 1: The snail climbs 5 feet... slides back down 4 feet, so it's back at the bottom... Day 4: 5 feet (total: 20 feet). The snail reaches the top on Day 4. **Answer: 4 days.** > 

*(Failure: Hallucinated a pattern and failed basic arithmetic tracking.)*

**Thinking V2 (Finetuned)**
> `<thinking>` 
> Day 1: Climbs 5, slips 4 = 1ft. Day 2: Climbs 5, slips 4 = 2ft... [Step-by-step tracking]... Day 15: Climbs 5, slips 4 = 15ft. Day 16: Climbs 5, hits 20ft and escapes. 
> `</thinking>`

> `<response>` 
> **16 days.**
> `</response>` 

*(Success: Correctly identified that the snail does not slide back once it reaches the top.)*


## How to Use the Code

### 1. The Prompt Format (Critical)
To trigger the reasoning engine, you **must** prime the model by leaving the `<thinking>` tag open at the end of your prompt.

```python
test_question = "A snail is at the bottom of a 20-foot well..."

# Prime the assistant response with the thinking tag
prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{test_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<thinking>\n"
```

### 2. Running Inference via Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "eelixir/llama3-8b-thinking-v2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

---

## Repository Contents
* `llama3_thinking_v2.ipynb`: The complete Google Colab notebook.
* `chain_of_thought_dataset.jsonl`: The 475 training examples. 

## Acknowledgments
Built utilizing the [Unsloth](https://github.com/unslothai/unsloth) library and the Hugging Face `trl` and `peft` ecosystems.