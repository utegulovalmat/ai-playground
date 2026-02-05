# LLM Examples Collection

A comprehensive, hands-on collection organized into 4 progressive learning phases.

## 📁 Structure

```
examples/
├── phase1_foundation/      # LLM API basics
│   ├── 01_openai_basics.py
│   ├── 02_anthropic_basics.py
│   ├── 03_gemini_basics.py
│   ├── 04_langchain_basics.py
│   ├── 05_pydantic_ai_basics.py
│   ├── 06_langgraph_basics.py
│   ├── 07_embeddings_basics.py
│   └── README.md
├── phase2_rag/            # RAG & Vector DBs
│   ├── 01_chromadb_basics.py
│   ├── 02_pinecone_basics.py
│   ├── 03_qdrant_basics.py
│   ├── 04_faiss_basics.py
│   ├── 05_vector_db_comparison.py
│   ├── 06_document_qa_system.py
│   └── README.md
├── phase3_production/     # Production APIs & Demos
│   ├── 01_fastapi_rest_api.py
│   ├── 02_gradio_demos.py
│   └── README.md
└── phase4_advanced/       # Advanced Techniques
    ├── 01_function_calling.py
    ├── 02_prompt_optimization.py
    └── README.md
```

## 🚀 Quick Start

### 1. Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies by Phase

**Phase 1 (Start here!):**
```bash
uv pip install openai anthropic google-genai langchain langgraph langchain-google-genai pydantic-ai cohere sentence-transformers scikit-learn
```

**Phase 2:**
```bash
uv pip install chromadb sentence-transformers pypdf pinecone-client qdrant-client faiss-cpu cohere scikit-learn
```

**Phase 3:**
```bash
uv pip install fastapi uvicorn gradio
```

**All phases:**
```bash
uv pip install -r requirements.txt
```

### 3. Set API Key
```bash
export GEMINI_API_KEY='your-key-here'
# or OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 4. Run Examples
```bash
# Phase 1
python examples/phase1_foundation/03_gemini_basics.py

# Phase 2
python examples/phase2_rag/01_chromadb_basics.py

# Phase 3
python examples/phase3_production/02_gradio_demos.py

# Phase 4
python examples/phase4_advanced/01_function_calling.py
```

## 📚 Learning Phases

### [Phase 1: Foundation](phase1_foundation/) ⭐
**Master LLM API basics** (7 files)

Learn: API authentication, conversations, streaming, function calling, embeddings

**Time:** 1-2 weeks

### [Phase 2: RAG & Vector DBs](phase2_rag/) ⭐⭐
**Build document Q&A systems** (6 files)

Learn: Embeddings, vector search, RAG, conversation memory, vector databases

**Time:** 1-2 weeks

### [Phase 3: Production](phase3_production/) ⭐⭐⭐
**Deploy production apps** (2 files)

Learn: REST APIs, async processing, streaming, UI demos

**Time:** 1-2 weeks

### [Phase 4: Advanced](phase4_advanced/) ⭐⭐⭐⭐
**Master advanced techniques** (2 files)

Learn: Multi-agent systems, prompt optimization, chain-of-thought

**Time:** 2+ weeks

## 🎯 Recommended Path

1. **Week 1-2:** Complete Phase 1, build a simple chatbot
2. **Week 3-4:** Complete Phase 2, build PDF Q&A system
3. **Week 5-6:** Complete Phase 3, deploy a demo
4. **Week 7+:** Complete Phase 4, build multi-agent system

## 💡 Tips

- **Start with Phase 1** - Don't skip ahead
- **Run every example** - Don't just read the code
- **Modify and experiment** - Change parameters to learn
- **Build projects** - Apply concepts immediately
- **Read phase READMEs** - Each has specific guidance

## 🔗 Resources

- [Main README](../README.md) - Project overview
- [UV Guide](../UV_GUIDE.md) - Package manager
- [AI Engineer Toolkit](../AI_ENGINEER_TOOLKIT.md) - Complete tool reference

## 📝 File Naming Convention

Files are numbered for recommended order:
- `01_*.py` - Start here
- `02_*.py` - Then this
- etc.

Each phase folder has its own README with detailed information.

---

**Ready to start?** Go to [`phase1_foundation/`](phase1_foundation/) and begin with `01_openai_basics.py` or `03_gemini_basics.py`!
