from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "trained_model.pkl"

# Standard feature order used by scikit-learn's Iris dataset.
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
SHORT = {f: f.replace(" (cm)", "") for f in FEATURES}
SPECIES = ["Iris setosa", "Iris versicolor", "Iris virginica"]

GREEN = "#2F4A3A"
SAGE = "#9DB49A"
OCHRE = "#B9893F"
CHARCOAL = "#2B2B2B"
OFFWHITE = "#F6F5F0"
COLORS = {"Iris setosa": SAGE, "Iris versicolor": GREEN, "Iris virginica": OCHRE}

DESCRIPTIONS = {
    "Iris setosa": "Has the smallest petals of the three species and is clearly "
    "separated from the others in the measurements.",
    "Iris versicolor": "Has medium-sized petals. Its measurements overlap with "
    "virginica, so these two are the harder pair to tell apart.",
    "Iris virginica": "Generally has the largest petals and sepals of the three species.",
}

# Allowed input range for the form (wider than the range seen in the dataset).
LIMITS = {
    "sepal length (cm)": (4.0, 8.5),
    "sepal width (cm)": (1.5, 5.0),
    "petal length (cm)": (0.5, 7.5),
    "petal width (cm)": (0.0, 3.0),
}
WIDGET_KEYS = {f: "in_" + SHORT[f].replace(" ", "_") for f in FEATURES}


# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Iris Flower Classification", layout="centered")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=Source+Sans+3:wght@400;600&display=swap');
    .stApp {{ background: {OFFWHITE}; color: {CHARCOAL};
              font-family: 'Source Sans 3', 'Segoe UI', sans-serif; }}
    .block-container {{ max-width: 1000px; padding-top: 2.5rem; }}
    h1, h2, h3 {{ font-family: 'Source Serif 4', Georgia, serif; color: {GREEN};
                  font-weight: 600; letter-spacing: 0; }}
    h1 {{ font-size: 2.2rem; }}
    h2, h3 {{ font-size: 1.4rem; margin-top: 0.5rem; }}
    [data-testid="stSidebar"] {{ background: #E4E8DC; border-right: 1px solid #CBD2C0; }}
    [data-testid="stBaseButton-primary"] {{ background: {GREEN}; border-color: {GREEN}; }}
    .result {{ border-left: 4px solid {GREEN}; padding: 0.3rem 0 0.3rem 1rem;
               margin: 0.5rem 0 1rem 0; color: {CHARCOAL}; }}
    .result .name {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.6rem;
                     color: {GREEN}; font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    """Standard Iris dataset from scikit-learn, with readable species names."""
    iris = load_iris(as_frame=True)
    df = iris.data[FEATURES].copy()
    names = ["Iris " + iris.target_names[t] for t in iris.target]
    df["species"] = pd.Categorical(names, categories=SPECIES, ordered=True)
    return df


@st.cache_resource
def load_model(path, modified_time):
    # joblib.load reads both joblib files and plain pickle files.
    # modified_time is only here so the cache refreshes if the file is replaced.
    return joblib.load(path)


def get_model():
    """Load the saved model, or show a clear error and stop the page."""
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Place `trained_model.pkl` in the same folder as "
            f"`app.py` (expected at `{MODEL_PATH}`) and reload the page."
        )
        st.stop()
    try:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as exc:
        st.error(
            f"Could not load `trained_model.pkl`: {exc}\n\n"
            "Check that the file is a valid pickle/joblib file and that the installed "
            "scikit-learn version matches the one used to train the model."
        )
        st.stop()
    if not hasattr(model, "predict"):
        st.error("The loaded object has no `predict()` method, so it cannot be used as a classifier.")
        st.stop()
    return model


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def species_name(label):
    """Map a model label (int, numeric string or species string) to a species name."""
    if isinstance(label, (int, float, np.integer, np.floating)):
        if float(label).is_integer() and 0 <= int(label) < len(SPECIES):
            return SPECIES[int(label)]
        return None
    key = str(label).lower().replace("iris", "").replace("-", "").replace("_", "").strip()
    if key.isdigit() and int(key) < len(SPECIES):
        return SPECIES[int(key)]
    for name in SPECIES:
        if name.lower().replace("iris", "").strip() == key:
            return name
    return None


def build_input(model, values):
    """Arrange the four values in the feature order the model was trained with."""
    n_expected = getattr(model, "n_features_in_", len(FEATURES))
    if n_expected != len(FEATURES):
        raise ValueError(f"The model expects {n_expected} features, but this app provides 4.")
    row = pd.DataFrame([list(values)], columns=FEATURES)
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return row.to_numpy()  # model was trained on a plain array
    names = list(names)
    if set(names) == set(FEATURES):
        return row[names]  # same names: reorder to match training
    # Different column names: assume the standard Iris order.
    return pd.DataFrame([list(values)], columns=names)


def run_prediction(model, values):
    x = build_input(model, values)
    label = model.predict(x)[0]
    result = {
        "values": tuple(values),
        "label": label,
        "species": species_name(label),
        "probs": None,
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        classes = list(model.classes_)
        probs = pd.DataFrame(
            {"Species": [species_name(c) or str(c) for c in classes], "Probability": proba}
        )
        if set(probs["Species"]) == set(SPECIES):
            probs = probs.set_index("Species").loc[SPECIES].reset_index()
        result["probs"] = probs
    return result


def label_check(model):
    """Predict each species' mean measurements and see which species the labels map to.

    This only checks that the label mapping looks consistent. It is not an accuracy measure.
    """
    means = load_data().groupby("species", observed=True)[FEATURES].mean().loc[SPECIES]
    rows = []
    for species, vals in means.iterrows():
        label = model.predict(build_input(model, tuple(vals)))[0]
        mapped = species_name(label)
        rows.append(
            {
                "Mean measurements of": species,
                "Model label": str(label),
                "Mapped to": mapped or "unrecognised",
                "Consistent": mapped == species,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def style(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CHARCOAL, size=13),
        title_font=dict(size=15, color=CHARCOAL),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#DDE1D5", zeroline=False)
    fig.update_yaxes(gridcolor="#DDE1D5", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------------

def page_overview():
    df = load_data()
    counts = df["species"].value_counts()
    means = df.groupby("species", observed=True)[FEATURES].mean()

    st.title("Iris Flower Classification")
    st.write(
        "The Iris dataset records four measurements for each of 150 flowers from three "
        "species of Iris. It was introduced by the statistician and biologist R. A. Fisher "
        "in 1936, using measurements collected by Edgar Anderson, and it is a standard "
        "first example for classification. This app uses a trained model to predict a "
        "flower's species from its measurements."
    )

    st.subheader("The three species")
    for col, sp in zip(st.columns(3), SPECIES):
        with col:
            st.markdown(f"**{sp}**")
            st.write(DESCRIPTIONS[sp])
            st.caption(
                f"{counts[sp]} samples. Mean petal length "
                f"{means.loc[sp, 'petal length (cm)']:.1f} cm, mean petal width "
                f"{means.loc[sp, 'petal width (cm)']:.1f} cm."
            )

    st.subheader("Input features")
    explain = {
        "sepal length (cm)": "Length of the sepal, the outer leaf-like part that encloses the bud.",
        "sepal width (cm)": "Width of the sepal at its widest point.",
        "petal length (cm)": "Length of the petal.",
        "petal width (cm)": "Width of the petal at its widest point.",
    }
    features = pd.DataFrame(
        {
            "Feature": [f.capitalize() for f in FEATURES],
            "What it measures": [explain[f] for f in FEATURES],
            "Range in dataset": [f"{df[f].min():.1f} to {df[f].max():.1f} cm" for f in FEATURES],
        }
    ).set_index("Feature")
    st.table(features)

    st.subheader("How a prediction is made")
    st.markdown(
        "1. On **Test New Samples**, enter the four measurements in centimetres, or load an example from the dataset.\n"
        "2. The app arranges the values in the feature order the model was trained with: "
        "sepal length, sepal width, petal length, petal width.\n"
        "3. The values are passed to the model in `trained_model.pkl`, which returns a species.\n"
        "4. If the model supports it, the app also shows the probability it assigns to each of the three species."
    )
    st.caption(
        "Model accuracy is not reported here. The app has no evaluation data for the saved model."
    )

    st.subheader("How the measurements differ by species")
    long = means.reset_index().melt(id_vars="species", var_name="Feature", value_name="Mean (cm)")
    long["Feature"] = long["Feature"].map(SHORT)
    fig = px.bar(
        long,
        x="Feature",
        y="Mean (cm)",
        color="species",
        barmode="group",
        color_discrete_map=COLORS,
        category_orders={"species": SPECIES},
    )
    fig.update_xaxes(title_text="")
    show(style(fig, 340))
    st.caption(
        "Mean of each measurement per species. Petal measurements differ much more between "
        "species than sepal measurements do."
    )


# ---------------------------------------------------------------------------
# Page 2: Data Summary
# ---------------------------------------------------------------------------

def page_data_summary():
    df = load_data()

    st.title("Data Summary")
    st.write(
        "This page uses the standard Iris dataset bundled with scikit-learn "
        "(`sklearn.datasets.load_iris`). The data used to train `trained_model.pkl` is not "
        "available to this app, so the data shown here may not be identical to the model's "
        "training data."
    )

    st.subheader("Dataset overview")
    n_per_class = df["species"].value_counts()
    st.write(
        f"The dataset has **{len(df)} samples**, **{len(FEATURES)} numeric features** and "
        f"**{df['species'].nunique()} target classes** ({n_per_class.min()} samples per class). "
        "Each row is one flower: four measurements in centimetres and its species."
    )
    overview = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Unit": "cm",
            "Minimum": [df[f].min() for f in FEATURES],
            "Maximum": [df[f].max() for f in FEATURES],
        }
    ).set_index("Feature")
    st.table(overview)

    st.subheader("Data table")
    chosen = st.multiselect("Show species", SPECIES, default=SPECIES)
    filtered = df[df["species"].isin(chosen)]
    st.dataframe(filtered, height=300)
    st.caption(f"Showing {len(filtered)} of {len(df)} rows.")

    st.subheader("Statistical summary")
    group = st.selectbox("Species", ["All species"] + SPECIES)
    subset = df if group == "All species" else df[df["species"] == group]
    stats = subset[FEATURES].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats.columns = ["Mean", "Std. dev.", "Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max"]
    st.table(stats.round(2))

    st.subheader("Visualizations")

    counts = df["species"].value_counts().reindex(SPECIES).reset_index()
    counts.columns = ["Species", "Samples"]
    fig = px.bar(
        counts,
        x="Species",
        y="Samples",
        color="Species",
        color_discrete_map=COLORS,
        title="Samples per species",
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="")
    show(style(fig, 280))

    kind = st.radio("Feature distribution", ["Histogram", "Box plot"], horizontal=True)
    feature = st.selectbox("Feature", FEATURES, index=2)
    if kind == "Histogram":
        fig = px.histogram(
            df,
            x=feature,
            color="species",
            barmode="overlay",
            opacity=0.75,
            nbins=20,
            color_discrete_map=COLORS,
            category_orders={"species": SPECIES},
            title=f"Distribution of {SHORT[feature]}",
        )
        fig.update_yaxes(title_text="Samples")
    else:
        fig = px.box(
            df,
            x="species",
            y=feature,
            color="species",
            color_discrete_map=COLORS,
            category_orders={"species": SPECIES},
            title=f"{SHORT[feature].capitalize()} by species",
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="")
    show(style(fig))

    fig = px.scatter(
        df,
        x="petal length (cm)",
        y="petal width (cm)",
        color="species",
        color_discrete_map=COLORS,
        category_orders={"species": SPECIES},
        opacity=0.85,
        title="Petal length vs petal width",
    )
    fig.update_traces(marker=dict(size=8))
    show(style(fig))

    corr = df[FEATURES].corr().rename(index=SHORT, columns=SHORT)
    fig = px.imshow(
        corr.round(2),
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale=[[0, OCHRE], [0.5, OFFWHITE], [1, GREEN]],
        title="Correlation between features",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="r"))
    show(style(fig, 420))
    st.caption(
        f"Petal length and petal width have a correlation of "
        f"{corr.loc['petal length', 'petal width']:.2f}."
    )


# ---------------------------------------------------------------------------
# Page 3: Test New Samples
# ---------------------------------------------------------------------------

def default_values():
    df = load_data()
    return {f: round(float(df[f].median()), 1) for f in FEATURES}


def init_inputs():
    """Create input values on first use and restore them after visiting another page."""
    defaults = default_values()
    for f in FEATURES:
        key = WIDGET_KEYS[f]
        if key not in st.session_state:
            st.session_state[key] = st.session_state.get("saved_" + key, defaults[f])


def set_inputs(values):
    for f in FEATURES:
        key = WIDGET_KEYS[f]
        st.session_state[key] = float(values[f])
        st.session_state["saved_" + key] = float(values[f])


def reset_inputs():
    set_inputs(default_values())
    st.session_state.pop("result", None)
    st.session_state.pop("example_note", None)


def load_example():
    df = load_data()
    choice = st.session_state.get("example_species", "Any species")
    pool = df if choice == "Any species" else df[df["species"] == choice]
    row = pool.sample(1).iloc[0]
    set_inputs(row)
    st.session_state["example_note"] = (
        f"Loaded dataset row {row.name}, listed as {row['species']}."
    )


def current_values():
    return tuple(float(st.session_state[WIDGET_KEYS[f]]) for f in FEATURES)


def validate(values):
    problems = []
    for f, v in zip(FEATURES, values):
        lo, hi = LIMITS[f]
        if v is None or np.isnan(v):
            problems.append(f"{f}: enter a number.")
        elif not lo <= v <= hi:
            problems.append(f"{f}: enter a value between {lo} and {hi}.")
    return problems


def outside_dataset_range(values):
    df = load_data()
    return [
        f"{SHORT[f]} ({v:.1f} cm)"
        for f, v in zip(FEATURES, values)
        if not df[f].min() <= v <= df[f].max()
    ]


def show_result(result):
    species = result["species"]
    if species is None:
        st.warning(
            f"The model returned the label {result['label']!r}, which could not be matched "
            "to one of the three Iris species."
        )
        return

    st.markdown(
        f'<div class="result">Predicted species<br><span class="name">{species}</span></div>',
        unsafe_allow_html=True,
    )
    st.write(f"The model classified these measurements as {species}. {DESCRIPTIONS[species]}")

    unusual = outside_dataset_range(result["values"])
    if unusual:
        st.info(
            "Outside the range seen in the Iris dataset: " + ", ".join(unusual) + ". "
            "Treat this prediction with extra caution."
        )

    probs = result["probs"]
    if probs is None:
        st.caption("This model does not provide class probabilities (no `predict_proba()`).")
        return
    fig = px.bar(
        probs,
        x="Probability",
        y="Species",
        orientation="h",
        color="Species",
        color_discrete_map=COLORS,
        text=probs["Probability"].map(lambda p: f"{p:.1%}"),
        title="Model probability for each species",
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="")
    fig.update_yaxes(title_text="", autorange="reversed")
    fig.update_traces(textposition="outside", cliponaxis=False)
    show(style(fig, 240))
    st.caption(
        "These are the model's own estimates for each class. They do not guarantee that the "
        "prediction is correct."
    )


def show_model_details(model, check):
    with st.expander("Model details and label check"):
        st.write(f"Model type: `{type(model).__name__}`")
        steps = getattr(model, "steps", None)
        if steps:
            st.write("Pipeline steps: " + ", ".join(f"`{n}` ({type(s).__name__})" for n, s in steps))
        st.write(
            "Probabilities available: "
            + ("yes" if hasattr(model, "predict_proba") else "no")
        )
        classes = getattr(model, "classes_", None)
        if classes is not None:
            mapping = pd.DataFrame(
                {
                    "Model label": [str(c) for c in classes],
                    "Mapped to": [species_name(c) or "unrecognised" for c in classes],
                }
            )
            st.dataframe(mapping, hide_index=True)
            if all(isinstance(c, (int, np.integer)) for c in classes):
                st.caption(
                    "Integer labels are assumed to follow scikit-learn's Iris order: "
                    "0 = setosa, 1 = versicolor, 2 = virginica."
                )
        if check is not None:
            st.write("Consistency check: the mean measurements of each species, run through the model.")
            st.dataframe(check, hide_index=True)
            st.caption("This checks the label mapping only. It is not a measure of accuracy.")


def page_test():
    st.title("Test New Samples")
    st.write(
        "Enter the four measurements of a flower and press **Predict Species**. "
        "No prediction runs until you press the button."
    )
    model = get_model()
    init_inputs()
    df = load_data()

    try:
        check = label_check(model)
    except Exception:
        check = None
    if check is not None and not check["Consistent"].all():
        st.warning(
            "Label check: the model does not classify the mean measurements of every species "
            "as that species. Open 'Model details and label check' below to see how its "
            "labels were mapped."
        )

    with st.expander("Start from an example in the dataset"):
        st.selectbox("Sample from", ["Any species"] + SPECIES, key="example_species")
        st.button("Load example", on_click=load_example)
        if "example_note" in st.session_state:
            st.caption(st.session_state["example_note"])

    left, right = st.columns(2)
    for col, feats in ((left, FEATURES[:2]), (right, FEATURES[2:])):
        with col:
            for f in feats:
                lo, hi = LIMITS[f]
                st.number_input(
                    f.capitalize(),
                    min_value=lo,
                    max_value=hi,
                    step=0.1,
                    format="%.1f",
                    key=WIDGET_KEYS[f],
                    help=f"Range in the dataset: {df[f].min():.1f} to {df[f].max():.1f} cm",
                )
    for f in FEATURES:  # remember the values if the user visits another page
        st.session_state["saved_" + WIDGET_KEYS[f]] = st.session_state[WIDGET_KEYS[f]]

    predict_col, reset_col, _ = st.columns([1, 1, 2])
    predict = predict_col.button("Predict Species", type="primary")
    reset_col.button("Reset inputs", on_click=reset_inputs)

    values = current_values()
    if predict:
        problems = validate(values)
        if problems:
            st.session_state.pop("result", None)
            st.error("Please fix the inputs:\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            try:
                st.session_state["result"] = run_prediction(model, values)
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"The prediction failed: {exc}")

    result = st.session_state.get("result")
    if result is not None:
        st.divider()
        if result["values"] != values:
            st.caption("The inputs have changed since this prediction. Press Predict Species to update it.")
        show_result(result)

    show_model_details(model, check)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Data Summary": page_data_summary,
    "Test New Samples": page_test,
}

st.sidebar.title("Iris classifier")
selected = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
st.sidebar.caption("Dataset: scikit-learn `load_iris`. Model: `trained_model.pkl`.")
PAGES[selected]()