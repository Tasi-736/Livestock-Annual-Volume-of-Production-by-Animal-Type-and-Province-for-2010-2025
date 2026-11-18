import glob
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)

try:
    from imblearn.over_sampling import SMOTENC
except ImportError:
    SMOTENC = None

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
YEARS = list(range(2010, 2026))
YEAR_COLUMNS = [str(y) for y in YEARS]
ID_COLUMNS = ["Animal Type", "Geolocation"]
ACCURACY_THRESHOLD = 0.80
AUGMENT_FACTOR = 1.25
FEATURES = ["Year", "Animal_Encoded", "Province_Encoded"]


# ===== PIPELINE START (same steps as the Colab notebook) =====
@st.cache_data(show_spinner=False)
def load_province_long(raw_text):
    """Read the ARFF/CSV text and return one row per province, animal and year."""
    lines = raw_text.splitlines()
    marker = [i for i, ln in enumerate(lines) if ln.strip().lower() == "@data"]
    if marker:
        data_text = "\n".join(lines[marker[0] + 1:]).strip()
        data = pd.read_csv(io.StringIO(data_text), header=None,
                           names=YEAR_COLUMNS + ID_COLUMNS, quotechar="'",
                           skipinitialspace=True, dtype=str, engine="python")
    else:
        data = pd.read_csv(io.StringIO(raw_text), quotechar="'", dtype=str)
        data.columns = [str(c).strip().strip("'\"") for c in data.columns]
        rename = {}
        for col in data.columns:
            key = col.strip().lower().replace("_", " ")
            if key in ["animal type", "animal"]:
                rename[col] = "Animal Type"
            elif key in ["geolocation", "geography", "province"]:
                rename[col] = "Geolocation"
        data = data.rename(columns=rename)[YEAR_COLUMNS + ID_COLUMNS]

    for col in YEAR_COLUMNS:
        data[col] = pd.to_numeric(data[col].astype(str).str.strip().replace({"..": np.nan}),
                                  errors="coerce")
    data["Animal Type"] = data["Animal Type"].astype(str).str.strip().str.strip("'\"")
    data["Geolocation_Raw"] = data["Geolocation"].astype(str).str.strip().str.strip("'\"")
    data["Geolocation"] = data["Geolocation_Raw"].str.replace(r"^\.+", "", regex=True).str.strip()
    data = data[data["Animal Type"].isin(["Carabao", "Cattle", "Goat", "Hog"])].copy()

    # Province rows have exactly four leading dots (skips national, regional and city rows)
    mask = data["Geolocation_Raw"].str.match(r"^\.{4}(?!\.)", na=False)
    wide = data.loc[mask, YEAR_COLUMNS + ["Animal Type", "Geolocation"]]
    long = wide.melt(id_vars=["Animal Type", "Geolocation"], value_vars=YEAR_COLUMNS,
                     var_name="Year", value_name="Production")
    long["Year"] = long["Year"].astype(int)
    long = long.dropna(subset=["Production"])
    return long.sort_values(["Geolocation", "Animal Type", "Year"]).reset_index(drop=True)


def prepare_model_df(province_long):
    """Label High/Low by the median and encode animal and province."""
    median_val = province_long["Production"].median()
    df = province_long.copy()
    df["Class"] = (df["Production"] >= median_val).astype(int)  # 0 = Low, 1 = High
    le_animal, le_province = LabelEncoder(), LabelEncoder()
    df["Animal_Encoded"] = le_animal.fit_transform(df["Animal Type"])
    df["Province_Encoded"] = le_province.fit_transform(df["Geolocation"])
    return df, median_val, le_animal, le_province


def model_specs():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6,
                                                random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                        max_depth=4, random_state=RANDOM_STATE),
    }


@st.cache_data(show_spinner=False)
def run_modeling(model_df, years_for_future):
    X, y = model_df[FEATURES], model_df["Class"]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=RANDOM_STATE, stratify=y_dev)

    metrics, aug_log, rocs, cms = [], [], {}, {}
    for name, spec in model_specs().items():
        base = clone(spec).fit(X_train, y_train)
        val_before = accuracy_score(y_val, base.predict(X_val))
        selected, rows_used, synthetic, val_after = base, len(X_train), 0, np.nan
        triggered = val_before < ACCURACY_THRESHOLD
        chosen, decision = False, "Not needed: validation accuracy is at least 80%."

        if triggered:
            counts = y_train.value_counts()
            if SMOTENC is None:
                decision = "Augmentation skipped: imbalanced-learn is not installed."
            elif counts.min() > 1:
                strategy = {int(k): int(np.ceil(v * AUGMENT_FACTOR)) for k, v in counts.items()}
                sampler = SMOTENC(categorical_features=[1, 2], sampling_strategy=strategy,
                                  random_state=RANDOM_STATE,
                                  k_neighbors=min(5, int(counts.min()) - 1))
                X_aug, y_aug = sampler.fit_resample(X_train, y_train)
                synthetic = len(X_aug) - len(X_train)
                aug_model = clone(spec).fit(X_aug, y_aug)
                val_after = accuracy_score(y_val, aug_model.predict(X_val))
                if val_after > val_before:
                    selected, rows_used, chosen = aug_model, len(X_aug), True
                    decision = "Augmented model selected: validation accuracy improved."
                else:
                    decision = "Augmentation tested; baseline kept because validation accuracy did not improve."
            else:
                decision = "Accuracy below 80%, but too few samples to generate synthetic rows."

        pred = selected.predict(X_test)
        prob = selected.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        rocs[name] = (fpr, tpr)
        cms[name] = confusion_matrix(y_test, pred, labels=[0, 1])
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)})
        aug_log.append({
            "Model": name, "Validation Accuracy Before": val_before,
            "Augmentation Triggered": "Yes" if triggered else "No",
            "Training Rows Before": len(X_train), "Synthetic Rows Created": synthetic,
            "Validation Accuracy After": val_after,
            "Augmented Model Selected": "Yes" if chosen else "No",
            "Final Training Rows Used": rows_used, "Decision": decision})

    comparison = (pd.DataFrame(metrics)
                  .sort_values(["Accuracy", "ROC-AUC", "F1-Score"], ascending=False)
                  .reset_index(drop=True))
    best = comparison.iloc[0]["Model"]

    # Final models are refit on all real records (no synthetic rows) for predictions
    final_models = {n: clone(s).fit(X, y) for n, s in model_specs().items()}
    sizes = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    return comparison, pd.DataFrame(aug_log), rocs, cms, best, final_models, sizes
# ===== PIPELINE END =====


# ---------- Load the data ----------
st.set_page_config(page_title="Livestock Production Analytics", layout="wide")
st.title("Livestock Production Analytics and Classification (2010-2025)")
st.caption("Annual volume of livestock production by animal type and province. "
           "Source: Philippine Statistics Authority.")

upload = st.sidebar.file_uploader("Upload the dataset (.arff or .csv)", type=["arff", "csv"])
if upload:
    raw_text = upload.getvalue().decode("utf-8-sig", errors="replace")
else:
    files = sorted(glob.glob("*.arff") + glob.glob("data/*.arff") + glob.glob("data/*.csv"))
    if not files:
        st.warning("Put the .arff dataset file in the same folder as app.py, or upload it in the sidebar.")
        st.stop()
    with open(files[0], encoding="utf-8-sig", errors="replace") as f:
        raw_text = f.read()

province_long = load_province_long(raw_text)
if province_long.empty:
    st.error("No province-level rows were found. Check that this is the annual province dataset.")
    st.stop()

model_df, median_val, le_animal, le_province = prepare_model_df(province_long)
st.sidebar.metric("Valid province-year records", f"{len(model_df):,}")
st.sidebar.metric("Provinces", model_df["Geolocation"].nunique())

tab1, tab2, tab3, tab4 = st.tabs(["Data visualization", "Descriptive analytics",
                                  "Prediction models", "2026 predictions"])

# ---------- 1. Data visualization ----------
with tab1:
    st.subheader("Cleaned dataset (one row per province, animal and year)")
    st.dataframe(province_long.head(200))

    st.subheader("Production trends by animal type")
    annual = province_long.pivot_table(index="Year", columns="Animal Type",
                                       values="Production", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(10, 5))
    annual.plot(ax=ax, marker="o")
    ax.set_ylabel("Summed provincial production")
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Production volume vs. year")
        fig, ax = plt.subplots()
        sns.scatterplot(data=province_long, x="Year", y="Production", hue="Animal Type",
                        alpha=0.6, palette="viridis", ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Correlation heatmap")
        corr = model_df[["Year", "Animal_Encoded", "Province_Encoded",
                         "Production", "Class"]].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)

# ---------- 2. Descriptive analytics ----------
with tab2:
    st.subheader("Descriptive statistics by animal type")
    summary = (province_long.groupby("Animal Type")["Production"]
               .agg(Count="count", Mean="mean", Median="median", Std_Dev="std",
                    Minimum="min", Maximum="max")
               .reset_index().sort_values("Mean", ascending=False))
    st.dataframe(summary.round(2))

    totals = (province_long.groupby("Animal Type", as_index=False)["Production"].sum()
              .sort_values("Production", ascending=False))
    totals["Share (%)"] = totals["Production"] / totals["Production"].sum() * 100
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Total production (bar)")
        fig, ax = plt.subplots()
        sns.barplot(data=totals, x="Animal Type", y="Production", hue="Animal Type",
                    palette="viridis", legend=False, ax=ax)
        ax.ticklabel_format(style="plain", axis="y")
        st.pyplot(fig)
    with c2:
        st.subheader("Share by animal type (pie)")
        fig, ax = plt.subplots()
        ax.pie(totals["Production"], labels=totals["Animal Type"], autopct="%1.1f%%",
               startangle=140, wedgeprops={"edgecolor": "white"})
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    p2025 = province_long[province_long["Year"] == 2025]
    with c3:
        st.subheader("Province-level distribution (2025)")
        fig, ax = plt.subplots()
        sns.boxplot(data=p2025, x="Animal Type", y="Production", showfliers=False, ax=ax)
        st.pyplot(fig)
    with c4:
        st.subheader("Top 10 hog-producing provinces (2025)")
        top = p2025[p2025["Animal Type"] == "Hog"].nlargest(10, "Production").sort_values("Production")
        fig, ax = plt.subplots()
        ax.barh(top["Geolocation"], top["Production"], color="#2B5F8A")
        st.pyplot(fig)

    st.subheader("High vs. Low production classes")
    counts = model_df["Class"].value_counts().reindex([0, 1], fill_value=0)
    st.caption(f"High production = at or above the median of {median_val:,.2f}.")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts.values, labels=["Low Production", "High Production"], autopct="%1.1f%%",
           startangle=140, colors=["#ff9999", "#66b3ff"], explode=(0.04, 0))
    st.pyplot(fig)

# ---------- 3. Models ----------
with st.spinner("Training the four models..."):
    comparison, aug_df, rocs, cms, best, final_models, sizes = run_modeling(model_df, 2026)

with tab3:
    st.subheader("Model comparison (untouched test set)")
    st.caption(f"Records: {sizes['train']:,} train, {sizes['val']:,} validation, {sizes['test']:,} test. "
               "If a model's validation accuracy is below 80%, synthetic training rows (SMOTENC) are "
               "tried and kept only if validation accuracy improves.")
    st.dataframe(comparison.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
        "F1-Score": "{:.2%}", "ROC-AUC": "{:.4f}"}))
    st.success(f"Highest accuracy on this test split: {best}")

    st.subheader("Conditional augmentation log")
    st.dataframe(aug_df.style.format({
        "Validation Accuracy Before": "{:.2%}",
        "Validation Accuracy After": lambda v: "N/A" if pd.isna(v) else f"{v:.2%}"}))

    st.subheader("ROC curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr) in rocs.items():
        auc = comparison.loc[comparison["Model"] == name, "ROC-AUC"].iloc[0]
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="No-skill baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.subheader("Confusion matrices")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        ConfusionMatrixDisplay(cm, display_labels=["Low", "High"]).plot(
            ax=ax, colorbar=False, values_format="d", cmap="Blues")
        ax.set_title(name)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Metrics comparison chart")
    long = comparison.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall",
                                                        "F1-Score", "ROC-AUC"],
                           var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="Model", y="Score", hue="Metric", palette="viridis", ax=ax)
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)

# ---------- 4. 2026 predictions ----------
with tab4:
    st.subheader("Predict High or Low production")
    st.caption(f"Uses the best model from the comparison ({best}), refit on all real records. "
               "This predicts a High/Low category, not an exact volume.")
    c1, c2, c3 = st.columns(3)
    year_in = c1.number_input("Year", min_value=2010, max_value=2040, value=2026)
    animal_in = c2.selectbox("Animal type", list(le_animal.classes_))
    prov_in = c3.selectbox("Province", list(le_province.classes_))
    row = pd.DataFrame([[year_in, le_animal.transform([animal_in])[0],
                         le_province.transform([prov_in])[0]]], columns=FEATURES)
    prob_high = float(final_models[best].predict_proba(row)[0][1])
    label = "High Production" if prob_high >= 0.5 else "Low Production"
    conf = prob_high if prob_high >= 0.5 else 1 - prob_high
    st.metric("Predicted class", label, f"{conf:.1%} confidence", delta_color="off")

    st.subheader("2026 predictions for every province and animal with a 2025 record")
    latest = province_long[province_long["Year"] == 2025].copy()
    X_future = pd.DataFrame({
        "Year": 2026,
        "Animal_Encoded": le_animal.transform(latest["Animal Type"]),
        "Province_Encoded": le_province.transform(latest["Geolocation"])})
    future = latest[["Geolocation", "Animal Type", "Production"]].rename(
        columns={"Production": "Production in 2025"}).reset_index(drop=True)
    for name, mdl in final_models.items():
        pred = mdl.predict(X_future)
        future[f"{name}"] = np.where(pred == 1, "High", "Low")
    st.dataframe(future)
    st.download_button("Download predictions (CSV)", future.to_csv(index=False),
                       "predicted_2026_high_low.csv", "text/csv")

st.caption("For teaching and exploratory analysis. Synthetic records are not official livestock measurements.")
