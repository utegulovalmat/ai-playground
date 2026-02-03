# LLM Examples Collection

A comprehensive, hands-on collection of examples for learning AI/LLM development, organized into 4 progressive phases.

## 📁 Project Structure

```
ai-playground/
├── examples/
│   ├── phase1_foundation/      # ⭐ Start here!
│   │   ├── 01_openai_basics.py
│   │   ├── 02_anthropic_basics.py
│   │   ├── 03_gemini_basics.py
│   │   ├── 04_langchain_basics.py
│   │   ├── 05_pydantic_ai_basics.py
│   │   └── README.md
│   ├── phase2_rag/            # ⭐⭐ RAG & Vector DBs
│   │   ├── 01_chromadb_basics.py
│   │   ├── 02_document_qa_system.py
│   │   └── README.md
│   ├── phase3_production/     # ⭐⭐⭐ Production APIs
│   │   ├── 01_fastapi_rest_api.py
│   │   ├── 02_gradio_demos.py
│   │   └── README.md
│   ├── phase4_advanced/       # ⭐⭐⭐⭐ Advanced
│   │   ├── 01_function_calling.py
│   │   ├── 02_prompt_optimization.py
│   │   └── README.md
│   └── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── AI_ENGINEER_TOOLKIT.md
├── UV_GUIDE.md
└── README.md (this file)
```

## 🚀 Quick Start

### 1. Install UV (Fast Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set Up Environment

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # macOS/Linux

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your key
# Get free API key from: https://aistudio.google.com
export GEMINI_API_KEY='your-key-here'
```

### 4. Run Your First Example

```bash
python examples/phase1_foundation/03_gemini_basics.py
```

## 📚 Learning Path

### Phase 1: Foundation (1-2 weeks)
**Master LLM API basics** - 5 examples

Start here! Learn API authentication, conversations, streaming, and function calling.

```bash
cd examples/phase1_foundation
python 03_gemini_basics.py
```

[→ Phase 1 Details](examples/phase1_foundation/)

### Phase 2: RAG & Vector DBs (1-2 weeks)
**Build document Q&A systems** - 2 examples

Learn embeddings, vector search, and retrieval-augmented generation.

```bash
cd examples/phase2_rag
python 01_chromadb_basics.py
```

[→ Phase 2 Details](examples/phase2_rag/)

### Phase 3: Production (1-2 weeks)
**Deploy production apps** - 2 examples

Create REST APIs and interactive demos.

```bash
cd examples/phase3_production
python 02_gradio_demos.py
```

[→ Phase 3 Details](examples/phase3_production/)

### Phase 4: Advanced (2+ weeks)
**Master advanced techniques** - 2 examples

Multi-agent systems and prompt optimization.

```bash
cd examples/phase4_advanced
python 01_function_calling.py
```

[→ Phase 4 Details](examples/phase4_advanced/)

## 🎯 What You'll Build

- ✅ **Week 1-2:** Simple chatbot (Phase 1)
- ✅ **Week 3-4:** PDF Q&A system (Phase 2)
- ✅ **Week 5-6:** Production API (Phase 3)
- ✅ **Week 7+:** Multi-agent system (Phase 4)

## 📖 Documentation

- **[examples/README.md](examples/README.md)** - Detailed guide for all phases
- **[AI_ENGINEER_TOOLKIT.md](AI_ENGINEER_TOOLKIT.md)** - Complete tool reference
- **[UV_GUIDE.md](UV_GUIDE.md)** - Package manager quick reference

## 🔧 Using UV

UV is 10-100x faster than pip:

```bash
# Install packages
uv pip install package-name

# Install from requirements
uv pip install -r requirements.txt

# Install by phase
uv pip install openai anthropic google-genai  # Phase 1
uv pip install chromadb sentence-transformers  # Phase 2
uv pip install fastapi uvicorn gradio          # Phase 3
```

## ✅ Verified & Tested

All 11 example files have been:
- ✓ Syntax checked
- ✓ Organized by difficulty
- ✓ Documented with comments
- ✓ Tested for correctness

Run the test suite:
```bash
python test_examples.py
```

## 💡 Tips for Success

1. **Follow the phases in order** - Each builds on the previous
2. **Run every example** - Don't just read the code
3. **Modify and experiment** - Change parameters to learn
4. **Build projects** - Apply concepts immediately
5. **Read the comments** - They contain important insights

## 🤝 Contributing

Improve examples or add new ones! Each should:
- Be self-contained and runnable
- Include comprehensive comments
- Follow best practices
- Handle errors gracefully

## 📝 License

Educational examples - use freely for learning and development.

---

**Ready to start?** Head to [`examples/phase1_foundation/`](examples/phase1_foundation/) and run your first example! 🚀
