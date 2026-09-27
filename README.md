# ABC Ltd — Customer Retention Predictor (Streamlit)

Two models trained on load and cached:
- **Logistic regression** — churn risk (Yes/No probability)
- **Linear regression** — predicted total billing (customer value)

## Files
- `app.py` — the Streamlit app (loads the pre-trained model, no training at runtime)
- `train_model.py` — run this once, offline, to (re)generate `model_bundle.pkl`
- `model_bundle.pkl` — the trained models, already included — app starts instantly
- `Telco-Customer-Churn.csv` — training data (only needed if you re-run train_model.py)
- `requirements.txt` — pinned dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## If you ever change the model (new features, different data)
```bash
python train_model.py   # regenerates model_bundle.pkl
streamlit run app.py     # picks up the new bundle automatically
```

## Deploy (GitHub + Streamlit Community Cloud) — see chat for full steps
1. Push this folder to a new GitHub repo.
2. Go to share.streamlit.io → New app → pick the repo, branch `main`, file `app.py`.
3. Deploy. Your app gets a public `*.streamlit.app` URL.
