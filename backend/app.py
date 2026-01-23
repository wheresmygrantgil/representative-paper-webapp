"""
Representative Paper Finder - Backend API

Provides REST endpoints for the frontend to:
1. Search authors via OpenAlex
2. Find representative papers using SPECTER embeddings

Deploy to Render with HF_TOKEN environment variable.
"""

import gc
import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import requests
from huggingface_hub import InferenceClient

# ============================================================================
# Configuration
# ============================================================================
MAX_PAPERS = 25
SPECTER_MODEL = "sentence-transformers/allenai-specter"
OPENALEX_BASE = "https://api.openalex.org"
CACHE_DIR = "embeddings/representative_papers"

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="Representative Paper Finder API",
    description="Find a researcher's most representative publication",
    version="1.0.0"
)

# CORS - Allow GitHub Pages and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://wheresmygrantgil.github.io",
        "*"  # For development - can restrict to just the above in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================
class SearchRequest(BaseModel):
    name: str


class AuthorResult(BaseModel):
    id: str
    label: str


class SearchResponse(BaseModel):
    authors: list[AuthorResult]


class FindRequest(BaseModel):
    author_id: str
    years: int = 5


class FindResponse(BaseModel):
    title: str
    abstract: str
    info: Optional[str] = None


# ============================================================================
# OpenAlex Client
# ============================================================================
class OpenAlexClient:
    """Client for OpenAlex API with rate limiting."""

    def __init__(self, email: str = None):
        self.session = requests.Session()
        if email:
            headers = {"User-Agent": f"mailto:{email}"}
        else:
            headers = {"User-Agent": "RepresentativePaperFinder/1.0"}
        self.session.headers.update(headers)
        self.last_request_time = 0
        self.min_interval = 0.1

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def fetch_json(self, endpoint: str, params: dict = None) -> dict:
        self._rate_limit()
        url = f"{OPENALEX_BASE}/{endpoint}"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                self.last_request_time = time.time()

                if response.status_code == 429:
                    wait_time = (2 ** attempt) * 1
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

        return {}


# Global client instance
openalex_client = OpenAlexClient()


# ============================================================================
# Helper Functions
# ============================================================================
def get_current_institution(author: dict) -> str:
    """Get the most recent institution from affiliations."""
    affiliations = author.get("affiliations") or []
    if not affiliations:
        return "Unknown institution"

    best_affiliation = None
    best_year = 0
    for aff in affiliations:
        years = aff.get("years") or []
        if years:
            max_year = max(years)
            if max_year > best_year:
                best_year = max_year
                best_affiliation = aff

    if best_affiliation:
        institution = best_affiliation.get("institution") or {}
        return institution.get("display_name", "Unknown institution")

    institution = affiliations[0].get("institution") or {}
    return institution.get("display_name", "Unknown institution")


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""

    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))

    words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words)


def fetch_publications_for_author(
    client: OpenAlexClient,
    author_id: str,
    years_back: int = 5,
    max_papers: int = None
) -> list[dict]:
    """Fetch recent publications for an author."""
    current_year = datetime.now().year
    start_year = current_year - years_back

    publications = []
    cursor = "*"

    while True:
        if max_papers and len(publications) >= max_papers:
            break

        per_page = 100
        if max_papers:
            per_page = min(100, max_papers - len(publications))

        params = {
            "filter": f"author.id:{author_id},publication_year:>{start_year - 1}",
            "sort": "publication_year:desc",
            "per_page": per_page,
            "cursor": cursor,
        }

        data = client.fetch_json("works", params)
        results = data.get("results", [])
        if not results:
            break

        publications.extend(results)

        if max_papers and len(publications) >= max_papers:
            break

        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor

    if max_papers:
        publications = publications[:max_papers]

    return publications


# ============================================================================
# Embedding Functions
# ============================================================================
def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_embedding(cache_dir: str, item_id: str):
    path = os.path.join(cache_dir, f"{item_id}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def save_embedding(cache_dir: str, item_id: str, vector: np.ndarray) -> None:
    try:
        _ensure_dir(cache_dir)
        np.save(os.path.join(cache_dir, f"{item_id}.npy"), vector)
    except Exception:
        pass


def encode_texts(texts: list[str]) -> np.ndarray:
    """Get embeddings from HuggingFace Inference API."""
    hf_token = os.environ.get("HF_TOKEN", "")
    client = InferenceClient(token=hf_token if hf_token else None)

    all_embeddings = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i + 8]
        for text in batch:
            embeddings = client.feature_extraction(text, model=SPECTER_MODEL)
            emb_array = np.array(embeddings)
            if emb_array.ndim == 2:
                pooled = np.mean(emb_array, axis=0)
            else:
                pooled = emb_array
            all_embeddings.append(pooled)

    return np.array(all_embeddings, dtype=np.float32)


def compute_medoid(vectors: np.ndarray) -> int:
    """Return the index of the medoid vector."""
    if len(vectors) == 0:
        raise ValueError("No vectors provided")
    v = vectors.astype(np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    sim = v @ v.T
    dist = 1.0 - sim
    return int(dist.sum(axis=1).argmin())


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/health")
def health_check():
    """Health check endpoint for keep-alive pings."""
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
def search_authors(request: SearchRequest):
    """Search for authors by name."""
    if not request.name or not request.name.strip():
        return SearchResponse(authors=[])

    try:
        params = {"search": request.name.strip(), "per_page": 10}
        data = openalex_client.fetch_json("authors", params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAlex API error: {str(e)}")

    results = data.get("results", [])
    authors = []

    for author in results:
        display_name = author.get("display_name", "Unknown")
        inst_name = get_current_institution(author)
        works_count = author.get("works_count", 0)
        author_id = author.get("id", "")

        label = f"{display_name} ({inst_name}, {works_count:,} works)"
        authors.append(AuthorResult(id=author_id, label=label))

    return SearchResponse(authors=authors)


@app.post("/api/find", response_model=FindResponse)
def find_representative_paper(request: FindRequest):
    """Find the most representative paper for a researcher."""
    if not request.author_id:
        raise HTTPException(status_code=400, detail="author_id is required")

    info_message = None

    # Fetch publications
    try:
        publications = fetch_publications_for_author(
            openalex_client, request.author_id, years_back=request.years
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching publications: {str(e)}")

    if not publications:
        raise HTTPException(status_code=404, detail="No publications found for this author in the selected time range.")

    # Extract valid publications with abstracts
    valid_pubs = []
    for pub in publications:
        title = pub.get("title", "")
        abstract_inverted = pub.get("abstract_inverted_index")
        abstract = reconstruct_abstract(abstract_inverted)

        if title and abstract:
            valid_pubs.append({"title": title, "abstract": abstract})

    del publications
    gc.collect()

    if not valid_pubs:
        raise HTTPException(status_code=404, detail="No publications with abstracts found.")

    # Limit to MAX_PAPERS
    total_found = len(valid_pubs)
    if total_found > MAX_PAPERS:
        valid_pubs = valid_pubs[:MAX_PAPERS]
        info_message = f"Note: Found {total_found} papers. Analyzing the {MAX_PAPERS} most recent due to computational limits."

    # Prepare texts for embedding
    texts = [f"{pub['title']}[SEP]{pub['abstract']}" for pub in valid_pubs]

    # Check cache and encode
    embeddings = []
    texts_to_encode = []
    text_indices = []

    for i, text in enumerate(texts):
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = load_embedding(CACHE_DIR, text_hash)
        if cached is not None:
            embeddings.append((i, cached))
        else:
            texts_to_encode.append(text)
            text_indices.append(i)

    if texts_to_encode:
        try:
            new_embeddings = encode_texts(texts_to_encode)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

        for idx, (text_idx, text) in enumerate(zip(text_indices, texts_to_encode)):
            vec = new_embeddings[idx]
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            save_embedding(CACHE_DIR, text_hash, vec)
            embeddings.append((text_idx, vec))

        del new_embeddings, texts_to_encode
        gc.collect()

    # Sort by original index and extract vectors
    embeddings.sort(key=lambda x: x[0])
    embedding_matrix = np.vstack([vec for _, vec in embeddings])

    del embeddings, texts
    gc.collect()

    medoid_idx = compute_medoid(embedding_matrix)
    representative = valid_pubs[medoid_idx]

    del embedding_matrix
    gc.collect()

    return FindResponse(
        title=representative["title"],
        abstract=representative["abstract"],
        info=info_message
    )


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
