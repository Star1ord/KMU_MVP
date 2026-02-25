# Web UI for Inference Demo

Lightweight static frontend to upload a video, run models and visualize results.

How to use

1. Start the backend API (from repo root):

```bash
# ensure backend runs on http://localhost:8000
python -m uvicorn api.main:app --reload --port 8000
```

2. Serve the `web_ui` folder (or open `index.html` directly). Recommended to run a simple static server:

```bash
cd web_ui
# Python 3 built-in server
python -m http.server 5500
```

Open http://localhost:5500 in your browser.

Notes
- The UI posts to `/predict`, `/test/nlp` and `/predict/cv-audio` endpoints. Adjust `apiBase` in `app.js` if backend runs elsewhere.
- Chart.js is loaded from CDN. No build step required.