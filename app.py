"""A beginner-friendly iris species prediction app."""

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@st.cache_resource
def train_model():
    """Train once and evaluate on samples excluded from training."""
    iris = load_iris(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, stratify=iris.target, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model, iris, model.score(x_test, y_test), len(x_train), len(x_test)


st.set_page_config(page_title="Iris flower predictor", page_icon=":material/local_florist:")
st.title("Iris flower predictor")
st.write("Enter a flower's measurements to predict whether it is setosa, versicolor, or virginica.")
st.caption("A simple machine learning project using a Random Forest classifier.")

with st.form("measurements"):
    st.subheader("Flower measurements")
    st.caption("All measurements are in centimetres. Defaults describe a typical setosa flower.")
    left, right = st.columns(2)
    with left:
        sepal_length = st.number_input("Sepal length (cm)", 4.3, 7.9, 5.1, 0.1)
        sepal_width = st.number_input("Sepal width (cm)", 2.0, 4.4, 3.5, 0.1)
    with right:
        petal_length = st.number_input("Petal length (cm)", 1.0, 6.9, 1.4, 0.1)
        petal_width = st.number_input("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)
    submitted = st.form_submit_button("Predict species", type="primary")

model, iris, accuracy, train_count, test_count = train_model()

if submitted:
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=iris.feature_names,
    )
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    st.success(f"Predicted species: {iris.target_names[prediction].capitalize()}")
    st.bar_chart(
        pd.DataFrame(
            {"Species": iris.target_names[model.classes_], "Probability": probabilities}
        ).set_index("Species")
    )
    st.caption("Probabilities are model estimates, not guarantees of a correct prediction.")

with st.container(border=True):
    st.subheader("Model performance")
    st.metric("Held-out test accuracy", f"{accuracy:.1%}")
    st.caption(f"Trained on {train_count} flowers and evaluated on {test_count} separate flowers.")

with st.expander("How it works and sample data"):
    st.write(
        "The built-in iris dataset contains 150 labelled flowers. We split it into "
        "80% training data and 20% test data, keeping the species balanced. "
        "A Random Forest combines 100 decision trees to predict the species. "
        "The model is cached so it does not retrain after each interaction."
    )
    preview = iris.data.copy()
    preview["Species"] = iris.target.map(dict(enumerate(iris.target_names)))
    st.dataframe(preview.head(10), hide_index=True)
    st.caption("Educational demo: the small test set does not establish performance on all real flowers.")
