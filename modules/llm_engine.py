import os
import torch

from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
LORA_DIR = os.path.join(MODEL_DIR, "recomind_lora_adapter")

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# GPU configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@lru_cache(maxsize=1)
def load_llm():

    print("Loading Qwen model...")
    print("Device:", DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=DTYPE,
        low_cpu_mem_usage=True
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        LORA_DIR
    )

    model.to(DEVICE)
    model.eval()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Allocated GPU memory:",
            round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "GB"
        )

    print("Qwen + LoRA loaded successfully!")

    return tokenizer, model


def generate_item_explanation(
    query,
    item_type,
    item_title,
    metadata
):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary.")

    tokenizer, model = load_llm()

    metadata_text = "\n".join(
        f"{key}: {value}"
        for key, value in metadata.items()
        if value is not None and str(value).strip()
    )

    prompt = f"""
You are RecomMind AI, an intelligent recommendation assistant.

User preferences:
{query}

Recommended {item_type}:
{item_title}

Verified metadata:
{metadata_text}

Explain why this {item_type} may match the user's preferences.

Rules:
- Use only the provided metadata.
- Never invent facts.
- Explain relevance using the user's preferences.
- Keep the explanation concise (2-3 sentences).
- Do not repeat the metadata as a list.
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[0][
        inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


def generate_movie_explanation(
    query,
    movie_title,
    genres,
    rating
):
    return generate_item_explanation(
        query=query,
        item_type="movie",
        item_title=movie_title,
        metadata={
            "Genres": genres,
            "Average rating": f"{rating:.2f}/5"
        }
    )


def explain_recommendations(query, recommendations):

    if recommendations.empty:
        return []

    explanations = []

    for _, movie in recommendations.iterrows():

        explanation = generate_movie_explanation(
            query,
            movie["title"],
            movie["genre_text"],
            float(movie["average_rating"])
        )

        explanations.append({
            "movie_id": int(movie["movie_id"]),
            "title": movie["title"],
            "genres": movie["genre_text"],
            "rating": float(movie["average_rating"]),
            "explanation": explanation
        })

    return explanations