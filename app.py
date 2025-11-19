# app.py
from flask import Flask, request, render_template, jsonify
from bs4 import BeautifulSoup
import requests
import re
import heapq
import os

# Optional HF model imports: loaded lazily if available
HF_AVAILABLE = False
HF_MODEL = None
HF_TOKENIZER = None

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    # Use a small-ish model that's reasonable for local CPU: 'sshleifer/distilbart-cnn-12-6'
    MODEL_NAME = os.getenv("HF_MODEL_NAME", "sshleifer/distilbart-cnn-12-6")
    HF_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    HF_MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    HF_AVAILABLE = True
    print("HF model loaded:", MODEL_NAME)
except Exception as e:
    print("HF model not available (will use TextRank).", str(e))
    HF_AVAILABLE = False

app = Flask(__name__, static_folder="static", template_folder="templates")

# -----------------------
# Extraction (readability-like)
# -----------------------
def extract_article_text(url, max_chars=20000):
    """
    Fetches URL and heuristically extracts the main article text by
    selecting the container with the most <p> text.
    """
    resp = requests.get(url, timeout=10, headers={"User-Agent": "ArticleSumm/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # remove scripts/styles
    for t in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
        t.decompose()

    candidates = soup.find_all(["article", "div", "section", "main"])
    best_text = ""
    best_len = 0

    # consider body as fallback
    if not candidates:
        body = soup.body
        if body:
            return " ".join(body.stripped_strings)[:max_chars]

    for c in candidates:
        ptexts = [p.get_text(" ", strip=True) for p in c.find_all("p")]
        text = " ".join(ptexts).strip()
        if len(text) > best_len:
            best_len = len(text)
            best_text = text

    cleaned = re.sub(r"\s+", " ", best_text).strip()
    return cleaned[:max_chars]

# -----------------------
# TextRank-like summarizer (simple, fast)
# -----------------------
def textrank_summarize(text, n_sentences=4):
    # split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= n_sentences:
        return text

    # word frequency
    words = re.findall(r"[a-zA-Z]+", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # score sentences by word frequency
    scores = {}
    for s in sentences:
        s_words = re.findall(r"[a-zA-Z]+", s.lower())
        score = sum(freq.get(w, 0) for w in s_words)
        scores[s] = score

    best = heapq.nlargest(n_sentences, scores, key=scores.get)
    return " ".join(best)

# -----------------------
# HF summarizer (if available)
# -----------------------
def hf_summarize(text, max_length=140, min_length=40):
    if not HF_AVAILABLE or HF_MODEL is None or HF_TOKENIZER is None:
        return None
    # tokenizer + model expects shorter inputs; truncate safely
    inputs = HF_TOKENIZER([text], max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = HF_MODEL.generate(**inputs, max_length=max_length, min_length=min_length, length_penalty=2.0)
    summary = HF_TOKENIZER.batch_decode(summary_ids, skip_special_tokens=True)[0]
    return summary

# -----------------------
# Flask routes
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    prefer_hf = data.get("prefer_hf", True)

    if not url:
        return jsonify({"error": "no url provided"}), 400

    try:
        article = extract_article_text(url)
        if not article:
            return jsonify({"error": "no article text extracted"}), 422

        summary = None
        if prefer_hf:
            try:
                summary = hf_summarize(article)
            except Exception:
                summary = None

        if not summary:
            summary = textrank_summarize(article, n_sentences=4)

        return jsonify({"article": article, "summary": summary, "used_hf": HF_AVAILABLE and prefer_hf and (summary is not None and HF_AVAILABLE)})
    except requests.HTTPError as he:
        return jsonify({"error": "failed to fetch url", "detail": str(he)}), 502
    except Exception as e:
        return jsonify({"error": "internal error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
