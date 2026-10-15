# HealthLens AI - Early Health Risk Analysis and Prediction Platform

An end-to-end data science project: EDA, four machine-learning models, honest evaluation, SHAP explanations and an interactive
Streamlit dashboard, built on the public **Pima Indians Diabetes Database**.

> **Educational and research prototype. Not a medical device.** It does not diagnose, treat or screen for any condition and must not
> replace advice from a healthcare professional. Predicted values are *model-estimated probabilities*, not a person's real risk.

## Features
- **Dataset Explorer** - browse records, validation, upload your own compatible CSV, source and limitations
- **Exploratory Analysis** - missing/impossible zeros, duplicates, distributions, outliers, Mann-Whitney + effect sizes, correlations, auto-generated findings
- **Health Risk Prediction** - validated inputs, "not measured" option, 4-model probabilities, reliability warnings, SHAP explanation in plain English
- **Model Comparison** - Logistic Regression, Decision Tree, Random Forest, XGBoost; metrics, ROC, confusion matrices, threshold analysis, cross-validation, bootstrap CIs, age-group check
- **Explainable AI** - SHAP global importance, beeswarm summary, per-record explanation, cross-model comparison
- **Reports** - HTML EDA report, CSV exports, one-click ZIP bundle
- **Docs** - full project report (generated from real results), methodology, 50 viva questions

## Project structure
```
HealthLens-AI/
├── app.py                      # entry point: navigation + Home page
├── pages/                      # Streamlit pages (1-7)
├── src/
│   ├── config.py               # paths, columns, seeds, colours
│   ├── data_loader.py          # load / validate / download dataset
│   ├── preprocessing.py        # leakage-free pipelines, split
│   ├── eda.py                  # statistics + charts + HTML report
│   ├── train_models.py         # tuning, CV, saving (CLI)
│   ├── evaluate_models.py      # metrics, ROC, thresholds, subgroups
│   ├── prediction.py           # input validation + prediction
│   ├── explainability.py       # SHAP
│   ├── generate_report.py      # builds docs/PROJECT_REPORT.md
│   └── ui.py, state.py, plotting.py
├── tests/                      # 41 pytest tests
├── data/  models/  docs/  reports/
├── requirements.txt            # flexible versions
└── requirements-tested.txt     # exact versions used during development
```

---
## 1. Run it in VS Code (step by step)

**Prerequisites:** Python 3.10 - 3.13 (developed and tested on 3.12; [python.org](https://www.python.org/downloads/); tick *Add Python to PATH* on Windows) and
[VS Code](https://code.visualstudio.com/) with the *Python* extension.

1. Unzip the project and open it: **File -> Open Folder -> HealthLens-AI**.
2. Open the terminal: **Terminal -> New Terminal** (make sure the path ends with `HealthLens-AI`).
3. Create a virtual environment

   Windows (PowerShell):
   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `venv\Scripts\activate.bat` in Command Prompt.

   macOS / Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Select the interpreter: press `Ctrl+Shift+P` -> *Python: Select Interpreter* -> choose the one inside `venv`.
5. Install dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   (If anything conflicts: `pip install -r requirements-tested.txt`.)
6. Get the dataset (already included; this command re-downloads it if you deleted it)
   ```bash
   python -m src.data_loader
   ```
7. Train the models (pre-trained files are included, so this is optional; run it to reproduce results)
   ```bash
   python -m src.train_models          # full run with nested CV, about 3 minutes
   python -m src.train_models --fast   # about 1 minute
   ```
8. Run the tests
   ```bash
   pip install pytest
   pytest
   ```
9. Start the app
   ```bash
   streamlit run app.py
   ```
   It opens at <http://localhost:8501>. Stop it with `Ctrl+C`.
10. (Optional) Regenerate the project report with your own numbers
    ```bash
    python -m src.generate_report
    ```

### Common problems
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run commands from the project root folder (the one containing `app.py`). |
| `streamlit` not recognised | Activate the venv first, or run `python -m streamlit run app.py`. |
| Port busy | `streamlit run app.py --server.port 8502` |
| Models fail to load after upgrading libraries | Just start the app - it retrains automatically - or run `python -m src.train_models`. |
| No internet for dataset | See `data/README.md` (manual download instructions). |



No paid API, key or credential is used anywhere. Do not commit private health data; uploaded CSVs stay in the user's session memory only.

## What was verified
All 41 automated tests pass, every page was exercised with Streamlit's `AppTest` harness (no exceptions, prediction and invalid-input flows included), and the
models and report were generated from a real training run. Visual appearance and Streamlit Cloud deployment were **not** tested in this build environment,
so check them on your machine and report anything that looks off.

## Citation of data
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset
of diabetes mellitus. *Proc. Annu. Symp. Comput. Appl. Med. Care*, 261-265. Data: NIDDK, via UCI/Kaggle.

## Author
Shubham Yadav

