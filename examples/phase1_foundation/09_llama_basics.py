"""
Llama 3 Local Inference Example
===============================
This example demonstrates running Meta's Llama 3 models locally using Transformers.
We use the Llama-3.2-1B-Instruct model as it is lightweight and runs on most consumer hardware.

Requirements:
- transformers
- torch
- accelerate
- HF_TOKEN environment variable (with access to Llama 3 models)

Hardware Notes:
- Mac (Apple Silicon): Uses MPS (Metal Performance Shaders) automatically via device_map="auto"
- NVIDIA GPU: Uses CUDA
- CPU: Fallback (slower)
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# Alternative for even smaller footprint: "meta-llama/Llama-3.2-1B" (base)
# Larger alternative: "meta-llama/Llama-3.1-8B-Instruct"

def check_env():
    if not os.environ.get("HF_TOKEN"):
        print("❌ Error: HF_TOKEN environment variable not set.")
        print("1. Get a token at https://huggingface.co/settings/tokens")
        print(f"2. Request access to {MODEL_ID}")
        print("3. Export it: export HF_TOKEN='your_token'")
        return False
    return True

def run_llama_inference():
    print(f"\n=== Loading {MODEL_ID} ===")
    
    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except OSError as e:
        if "401" in str(e):
            print("\n❌ Authorization Error: You likely don't have access to this gated model.")
            print(f"Please visit https://huggingface.co/{MODEL_ID} to accept the license.")
        else:
            print(f"\n❌ Error loading tokenizer: {e}")
        return

    # 2. Load Model
    # Use float16 for better performance on Mac/GPU
    # device_map="auto" will choose GPU/MPS if available
    print("Loading model weights... (this may take a minute)")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("✓ Model loaded successfully!")
    print(f"  Device: {model.device}")
    
    # 3. Create Chat Pipeline
    # Llama 3 uses a specific chat template (<|begin_of_text|><|start_header_id|>...)
    # The pipeline handles this automatically if we use the chat interface.
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 4. Generate Response
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant who answers concisely."},
        {"role": "user", "content": "Explain specific heat capacity in one sentence."}
    ]

    print("\n=== Generating Response ===")
    print(f"Input: {messages[-1]['content']}")
    
    # Apply chat template manually (optional, but good for understanding)
    # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print(f"Formatted Prompt: {prompt!r}")

    outputs = pipe(
        messages,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = outputs[0]["generated_text"][-1]["content"]
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    print("=" * 60)
    print("Llama 3 Local Inference")
    print("=" * 60)
    
    if check_env():
        run_llama_inference()
        
    print("\n" + "=" * 60)
    print("✓ Done")
