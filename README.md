# streamlitprojec

A simple machine learning project that predicts iris flower species from four measurements using a Random Forest classifier.

## Run on Windows

Open PowerShell in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 in your browser. Enter the sepal and petal measurements, then click **Predict species**.

If you already have the dependencies installed, run `python -m streamlit run app.py`.

## What the project does

- Uses scikit-learn's built-in iris dataset; no download or API key is required.
- Trains 100 decision trees on 120 flowers using a reproducible, stratified split.
- Shows the predicted species and estimated probabilities for all three species.
- Reports accuracy on 30 separate test flowers.
- Caches the model to avoid training again on every interaction.

Input ranges follow the dataset's observed measurements. This is an educational demo, and probabilities are estimates rather than guarantees.

`app.py` contains the interface and model training. `requirements.txt` lists dependencies.
