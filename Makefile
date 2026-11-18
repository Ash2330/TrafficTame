# Traffic Congestion Prediction Project — Makefile
# Usage (PowerShell): just run each target as: make <target>
# Or run steps manually using the commands below.
#
# For Windows without GNU Make, use the equivalent PowerShell commands
# listed in README.md Section 3.

PYTHON  = .venv\Scripts\python
PYTEST  = .venv\Scripts\pytest
JUPYTER = .venv\Scripts\jupyter
STREAMLIT = .venv\Scripts\streamlit

.PHONY: all env data preprocess features train notebooks test app clean

## Full reproducible pipeline
all: env data preprocess features train notebooks test

## 1. Create virtual environment and install dependencies
env:
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

## 2. Download dataset from UCI repository (requires internet access)
data:
	$(PYTHON) -c "from ucimlrepo import fetch_ucirepo; import pandas as pd; \
	  d = fetch_ucirepo(id=492); \
	  df = pd.concat([d.data.features, d.data.targets], axis=1); \
	  df.to_csv('data/raw/Metro_Interstate_Traffic_Volume.csv', index=False); \
	  print('Downloaded', len(df), 'rows')"

## 3. Preprocess raw data
preprocess:
	$(PYTHON) src/preprocess.py

## 4. Build feature engineering diagnostics (optional — train.py also runs this)
features:
	$(PYTHON) src/features.py

## 5. Train, evaluate and serialize all models
train:
	$(PYTHON) src/train.py

## 6. Regenerate all Jupyter notebooks from scripts
notebooks:
	$(PYTHON) scripts/create_eda_notebook.py
	$(PYTHON) scripts/create_features_notebook.py
	$(PYTHON) scripts/create_models_notebook.py

## 7. Run the full test suite
test:
	$(PYTEST) tests/ -v

## 8. Launch the Streamlit dashboard (opens browser at http://localhost:8501)
app:
	$(STREAMLIT) run app/streamlit_app.py

## 9. Remove Python caches (not data or models)
clean:
	if exist src\__pycache__ rmdir /s /q src\__pycache__
	if exist tests\__pycache__ rmdir /s /q tests\__pycache__
	if exist app\__pycache__ rmdir /s /q app\__pycache__
	if exist .pytest_cache rmdir /s /q .pytest_cache
