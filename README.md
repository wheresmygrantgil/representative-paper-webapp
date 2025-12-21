---
title: Representative Paper Finder
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Representative Paper Finder

Find a researcher's most representative publication using AI-powered semantic analysis.

## How it works

1. **Search** - Enter a researcher's name to find them in OpenAlex
2. **Select** - Choose the correct author from search results
3. **Find** - Select how many years to look back and click Find
4. **Result** - See their most representative paper (the medoid of their publication embeddings)

## Technology

- **OpenAlex API** - Free, open catalog of the world's scholarly works
- **SPECTER** - Scientific paper embeddings from AllenAI
- **Medoid calculation** - Finds the paper most similar to all others (the "center" of their research)

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860
