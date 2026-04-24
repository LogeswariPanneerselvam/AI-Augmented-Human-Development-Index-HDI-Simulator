import streamlit as st
import pandas as pd
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI-Augmented HDI Simulator", layout="wide")

# ==============================
# TITLE & DESCRIPTION
# ==============================
st.title("🌐 AI-Augmented Human Development Index (HDI) Simulator")

st.markdown("""
This tool compares **Traditional HDI** with an **AI-Augmented HDI** by including **Digital Readiness**.

### Instructions:
- Use the sliders on the left
- Values should be between **0 and 1**
- Observe how digital changes impact development
""")

st.caption("Note: This model is used for scenario analysis and interpretability, not exact prediction.")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_excel("final_output.xlsx")

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model.pkl")

X_cols = ["Health_Index","Education_Index","Income_Index","Digital_Index_norm"]

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("🔧 Input Parameters")

health = st.sidebar.slider("Health Index", 0.5, 1.0, 0.8)
education = st.sidebar.slider("Education Index", 0.5, 1.0, 0.7)
income = st.sidebar.slider("Income Index", 0.5, 1.0, 0.7)
digital = st.sidebar.slider("Digital Index", 0.0, 1.0, 0.5)

# ==============================
# CALCULATIONS
# ==============================

# Traditional HDI
hdi = (health * education * income) ** (1/3)

# AI-Augmented HDI (formula)
ai_hdi = (health * education * income * digital) ** (1/4)

# ML Prediction
input_df = pd.DataFrame([[health, education, income, digital]], columns=X_cols)
predicted = model.predict(input_df)[0]

# Impact
impact = ai_hdi - hdi

# ==============================
# RESULTS DISPLAY
# ==============================
st.subheader("📊 Results")

col1, col2, col3 = st.columns(3)

col1.metric("Traditional HDI", round(hdi, 3))
col2.metric("AI-HD (Formula)", round(ai_hdi, 3))
col3.metric("AI-HD (ML Prediction)", round(predicted, 3))

st.markdown("---")

# ==============================
# IMPACT ANALYSIS
# ==============================
st.subheader("📈 Impact of Digital Inclusion")

if impact > 0:
    st.success(f"Digital improvement increased HDI by {round(impact,3)}")
else:
    st.error(f"Lower digital readiness reduced HDI by {round(impact,3)}")

# ==============================
# FEATURE IMPORTANCE
# ==============================
st.subheader("🧠 Key Drivers")

importance = pd.Series(model.coef_, index=X_cols)
importance_sorted = importance.sort_values()

st.bar_chart(importance_sorted)

top_feature = importance.abs().idxmax()

st.info(f"Top Driving Factor: {top_feature}")

# ==============================
# INTERPRETATION
# ==============================
st.subheader("📝 Interpretation")

st.markdown("""
- Digital readiness shows the strongest influence in this model
- Higher digital access improves development outcomes
- Lower digital access can reduce overall HDI

⚠️ Limitations:
- Based on limited state-level data (~36 observations)
- Variables may be correlated
- Model used for analysis, not exact prediction
""")

# ==============================
# OPTIONAL DATA VIEW
# ==============================
with st.expander("📂 View Dataset"):
    st.dataframe(df)