"""
Hugging Face Ecosystem Basics
=============================
This example demonstrates how to use the Hugging Face ecosystem, including
Transformers pipelines, the Inference API, and manual model loading.

Requirements:
- transformers
- huggingface_hub
- torch

Best Practices:
- Use Pipelines for quick, easy inference
- Use InferenceClient for serverless inference (saves disk space)
- Use AutoModel/AutoTokenizer for fine-grained control
"""

import os
import sys

def check_dependencies():
    try:
        import transformers
        import huggingface_hub
        import torch
        print(f"Transformers version: {transformers.__version__}")
        print(f"Torch version: {torch.__version__}")
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        return False

def pipelines_example():
    """
    Demonstrates the simplest way to use models: Pipelines.
    Automates preprocessing, inference, and postprocessing.
    """
    print("\n=== Transformers Pipelines ===")
    from transformers import pipeline

    # 1. Sentiment Analysis (downloads a small model by default)
    print("\n1. Sentiment Analysis:")
    # explicit logic to avoid warning users about default models
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    
    texts = [
        "I love exploring new AI technologies!",
        "Debugging dependency conflicts is frustrating but necessary."
    ]
    
    results = sentiment_pipe(texts)
    for text, result in zip(texts, results):
        print(f"  Input: '{text}'")
        print(f"  Result: {result['label']} (Score: {result['score']:.4f})")

    # 2. Text Generation (using a small, fast model)
    print("\n2. Text Generation (GPT-2):")
    generator = pipeline("text-generation", model="gpt2")
    output = generator("The future of AI is", max_length=30, num_return_sequences=1, truncation=True)
    print(f"  Output: {output[0]['generated_text']}")


def inference_api_example():
    """
    Demonstrates using the Serverless Inference API.
    Runs models on Hugging Face infrastructure without downloading them.
    Requires HF_TOKEN for higher rate limits.
    """
    print("\n=== Serverless Inference API ===")
    from huggingface_hub import InferenceClient

    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        print("⚠️  HF_TOKEN not set. Rate limits may be strict.")
    
    # Initialize client
    client = InferenceClient(token=api_token)

    # Text Generation (using a high-quality open model)
    model_id = "meta-llama/Llama-2-7b-chat-hf" 
    # Note: Llama 2 might need access. Using a generic open model to be safe.
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    print(f"\nQuerying {model_id} via API...")
    try:
        response = client.text_generation(
            "Write a haiku about coding.",
            model=model_id,
            max_new_tokens=50
        )
        print(f"  Response:\n{response}")
    except Exception as e:
        print(f"  API Error: {e}")
        print("  (Authentication or model access might be required)")


def manual_loading_example():
    """
    Demonstrates loading Tokenizers and Models manually.
    Gives explicit control over the generation loop.
    """
    print("\n=== Manual Model Loading ===")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "gpt2" # Using GPT-2 for speed and size
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    input_text = "Python is a great language because"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate
    output_ids = model.generate(
        input_ids, 
        max_length=40, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"  Input: {input_text}")
    print(f"  Generated: {output_text}")


if __name__ == "__main__":
    if check_dependencies():
        print("=" * 60)
        pipelines_example()
        
        print("=" * 60)
        inference_api_example()
        
        print("=" * 60)
        manual_loading_example()
        
        print("=" * 60)
        print("✓ All examples completed!")
