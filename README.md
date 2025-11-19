This web application allows users to paste any article link and instantly receive a cleanly extracted version of the article along with an AI-generated summary. It’s built to be lightweight, privacy-friendly, and completely local, requiring no external APIs.

The app performs three key functions:

1. URL Input & Article Fetching

Users paste any URL into a simple web interface.
The backend fetches the page and removes scripts, ads, navigation, and other noise. A small Readability-style heuristic identifies the part of the page containing the densest paragraph content, producing a clean article body.

2. Intelligent Summarization (Two Modes)

The app supports two summarization engines:

Local ML Summarizer (if installed)

If the user has a small Hugging Face transformer model available locally (e.g., DistilBART), the app automatically uses it for higher-quality, abstractive summarization — entirely offline and without API calls.

Custom Algorithm (fallback)

If no ML model is available, the app switches to a fast, built-in TextRank-inspired summarizer that scores sentences and selects the most informative ones. This keeps the app functional even in minimal environments.

3. Clean, Responsive Web Interface

The frontend provides:

A simple URL input

An optional toggle to prefer the ML model

Separate sections for extracted article text and generated summary

Helpful status messages and error handling

Everything runs on a small Flask server with vanilla HTML/CSS/JS on the frontend, making it easy to deploy, modify, and understand.
