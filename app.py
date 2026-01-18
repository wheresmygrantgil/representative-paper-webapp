"""
Representative Paper Finder - Gradio Web App

Find a researcher's most representative publication using AI-powered semantic analysis.
Uses OpenAlex API for publication data and SPECTER embeddings for similarity.
"""

import hashlib
import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
import requests
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoTokenizer


# ============================================================================
# OpenAlex API Client
# ============================================================================
BASE_URL = "https://api.openalex.org"


class OpenAlexClient:
    """Client for OpenAlex API with rate limiting and error handling."""

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
        url = f"{BASE_URL}/{endpoint}"
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


# ============================================================================
# Publication Functions
# ============================================================================
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
# Embedding Cache (optional, for local deployment)
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
        pass  # Ignore cache errors in hosted environments


# ============================================================================
# Global State
# ============================================================================
onnx_session = None
tokenizer = None
client = None
CACHE_DIR = "embeddings/representative_papers"
MODEL_PATH = "model/model_quantized.onnx"
TOKENIZER_NAME = "sentence-transformers/allenai-specter"


def get_onnx_session():
    """Load the quantized ONNX model."""
    global onnx_session
    if onnx_session is None:
        onnx_session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
    return onnx_session


def get_tokenizer():
    """Load the HuggingFace tokenizer."""
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return tokenizer


def mean_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Apply mean pooling to get sentence embeddings."""
    # Expand attention mask for broadcasting
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    # Sum embeddings weighted by attention mask
    sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
    # Count non-masked tokens
    sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode texts to embeddings using ONNX model."""
    session = get_onnx_session()
    tok = get_tokenizer()

    # Tokenize
    inputs = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # Create token_type_ids if not present (BERT-style models need this)
    token_type_ids = inputs.get("token_type_ids", np.zeros_like(input_ids))

    # Run inference
    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    )

    # outputs[0] is last_hidden_state
    last_hidden_state = outputs[0]

    # Apply mean pooling
    embeddings = mean_pooling(last_hidden_state, attention_mask)

    return embeddings


def get_client():
    global client
    if client is None:
        client = OpenAlexClient()
    return client


# ============================================================================
# Core Functions
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


def search_authors(name: str):
    """Search for authors by name and return choices for dropdown."""
    if not name or not name.strip():
        return gr.update(choices=[], value=None)

    api = get_client()
    params = {"search": name.strip(), "per_page": 10}

    try:
        data = api.fetch_json("authors", params)
    except Exception:
        return gr.update(choices=[], value=None)

    results = data.get("results", [])
    if not results:
        return gr.update(choices=[], value=None)

    choices = []
    for author in results:
        display_name = author.get("display_name", "Unknown")
        inst_name = get_current_institution(author)
        works_count = author.get("works_count", 0)
        author_id = author.get("id", "")

        label = f"{display_name} ({inst_name}, {works_count:,} works)"
        choices.append((label, author_id))

    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def compute_medoid(vectors: np.ndarray) -> int:
    """Return the index of the vector that is the medoid."""
    if len(vectors) == 0:
        raise ValueError("No vectors provided")
    distances = cosine_distances(vectors)
    medoid_idx = distances.sum(axis=1).argmin()
    return int(medoid_idx)


def find_representative_paper(author_id: str, years: int):
    """Find the most representative paper for a researcher."""
    if not author_id:
        return "Please search and select an author first.", ""

    api = get_client()

    try:
        publications = fetch_publications_for_author(
            api, author_id, years_back=int(years)
        )
    except Exception as e:
        return f"Error fetching publications: {e}", ""

    if not publications:
        return "No publications found for this author in the selected time range.", ""

    valid_pubs = []
    texts = []
    for pub in publications:
        title = pub.get("title", "")
        abstract_inverted = pub.get("abstract_inverted_index")
        abstract = reconstruct_abstract(abstract_inverted)

        if title and abstract:
            valid_pubs.append({"title": title, "abstract": abstract})
            texts.append(f"{title}[SEP]{abstract}")

    if not valid_pubs:
        return "No publications with abstracts found.", ""

    # Check cache for each text
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

    # Encode all uncached texts in one batch
    if texts_to_encode:
        new_embeddings = encode_texts(texts_to_encode)
        for idx, (text_idx, text) in enumerate(zip(text_indices, texts_to_encode)):
            vec = new_embeddings[idx]
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            save_embedding(CACHE_DIR, text_hash, vec)
            embeddings.append((text_idx, vec))

    # Sort by original index and extract vectors
    embeddings.sort(key=lambda x: x[0])
    embeddings = np.vstack([vec for _, vec in embeddings])

    medoid_idx = compute_medoid(embeddings)
    representative = valid_pubs[medoid_idx]

    return representative["title"], representative["abstract"]


# ============================================================================
# UI Configuration
# ============================================================================
custom_css = """
.gradio-container {
    max-width: 600px !important;
    margin: auto !important;
}
.main-header {
    text-align: center;
    margin-bottom: 1rem;
}
.main-header h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
.step-header {
    font-weight: 600;
    color: #4f46e5;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    font=gr.themes.GoogleFont("Inter"),
)


# ============================================================================
# Visibility Handlers
# ============================================================================
def search_and_show(name: str):
    result = search_authors(name)
    has_results = result.get("choices") and len(result["choices"]) > 0
    if has_results:
        return result, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    return result, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


def confirm_author(author_id):
    if author_id:
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=True), gr.update(visible=False)


def show_loading():
    return gr.update(value="⏳ *Analyzing publications... This may take a moment.*", visible=True)


def show_search_loading():
    return gr.update(value="⏳ *Searching...*", visible=True)


def find_and_show(author_id: str, years: int):
    title, abstract = find_representative_paper(author_id, years)
    return gr.update(visible=False), title, abstract, gr.update(visible=False), gr.update(visible=True)


def reset_all():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        "",
        "",
    )


# ============================================================================
# Gradio Interface
# ============================================================================
with gr.Blocks(title="Representative Paper Finder", theme=theme, css=custom_css) as demo:

    gr.HTML("""
        <div class="main-header">
            <h1>📚 Representative Paper Finder</h1>
        </div>
    """)
    gr.Markdown(
        "<p class='subtitle'>Find a researcher's most representative publication</p>",
    )

    # Step 1: Search
    step1 = gr.Group(visible=True)
    with step1:
        gr.Markdown("<p class='step-header'>① Search Researcher</p>")
        name_input = gr.Textbox(
            label="Researcher Name",
            placeholder="e.g., Yann LeCun, Geoffrey Hinton...",
        )
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        search_status = gr.Markdown(visible=False)

    # Step 2: Select Author
    step2 = gr.Group(visible=False)
    with step2:
        gr.Markdown("<p class='step-header'>② Select Author</p>")
        author_dropdown = gr.Dropdown(
            label="Choose from search results",
            choices=[],
            interactive=True,
        )
        confirm_btn = gr.Button("→ Continue", variant="primary", size="lg")

    # Step 3: Configure & Find
    step3 = gr.Group(visible=False)
    with step3:
        gr.Markdown("<p class='step-header'>③ Find Representative Paper</p>")
        years_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Years to look back",
        )
        find_btn = gr.Button("✨ Find", variant="primary", size="lg")
        status_text = gr.Markdown(visible=False)

    # Step 4: Results
    step4 = gr.Group(visible=False)
    with step4:
        gr.Markdown("<p class='step-header'>📄 Result</p>")
        title_output = gr.Textbox(
            label="Paper Title",
            interactive=False,
            lines=2,
        )
        abstract_output = gr.Textbox(
            label="Abstract",
            interactive=False,
            lines=12,
        )
        reset_btn = gr.Button("🔄 Start Over", variant="secondary", size="lg")

    gr.Markdown(
        "<p style='text-align: center; color: #9ca3af; margin-top: 1.5rem; font-size: 0.8rem;'>"
        "Powered by OpenAlex & SPECTER"
        "</p>"
    )

    # Events
    search_btn.click(
        fn=show_search_loading,
        inputs=None,
        outputs=search_status,
    ).then(
        fn=search_and_show,
        inputs=name_input,
        outputs=[author_dropdown, step1, step2, search_status],
    )

    name_input.submit(
        fn=show_search_loading,
        inputs=None,
        outputs=search_status,
    ).then(
        fn=search_and_show,
        inputs=name_input,
        outputs=[author_dropdown, step1, step2, search_status],
    )

    confirm_btn.click(
        fn=confirm_author,
        inputs=author_dropdown,
        outputs=[step2, step3],
    )

    find_btn.click(
        fn=show_loading,
        inputs=None,
        outputs=status_text,
    ).then(
        fn=find_and_show,
        inputs=[author_dropdown, years_slider],
        outputs=[status_text, title_output, abstract_output, step3, step4],
    )

    reset_btn.click(
        fn=reset_all,
        inputs=None,
        outputs=[step1, step2, step3, step4, status_text, search_status, author_dropdown, title_output, abstract_output],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
