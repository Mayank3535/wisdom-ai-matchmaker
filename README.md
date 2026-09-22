# Wisdom AI MatchMaker

AI-Powered Client–Supplier Matchmaking Platform built for the Wisdom Group AI Intern evaluation.

## Features
- Client and supplier portals
- Persistent SQLite-style JSON storage for fast demo setup
- ML-based TF-IDF semantic similarity (not exact keyword matching)
- Hybrid weighted matching score
- Quantity, budget, category, location and delivery compatibility
- Explainable match score
- Ranked supplier recommendations
- Notifications
- Dashboard and demo data

## AI Matching Approach
The engine vectorizes client requirements and supplier offerings using TF-IDF with unigram and bigram features and calculates cosine similarity. This semantic text score is combined with structured business compatibility:

- Semantic similarity: 35%
- Category compatibility: 15%
- Quantity feasibility: 15%
- Budget compatibility: 15%
- Location compatibility: 10%
- Delivery compatibility: 10%

The resulting score is persisted with each match and displayed with a factor-by-factor explanation.

## Technology
- Python
- Flask
- scikit-learn
- HTML/CSS/JavaScript
- Chart.js
- JSON persistence for lightweight deployment/demo

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Demo
Use **Load Demo Data** on the dashboard, then inspect ranked matches. The sample data contains solar equipment and textile procurement scenarios.

## Project Structure
- `app.py` - Flask API and matching engine
- `static/index.html` - application UI
- `data.json` - local persistence
