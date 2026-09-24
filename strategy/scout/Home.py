import streamlit as st

st.set_page_config(
    page_title="FRC Scout AI",
    page_icon="robot",
    layout="wide",
)

st.title("FRC Scout AI")
st.caption("2026 Season — Powered by Ollama + Statbotics")

st.markdown("""
Welcome to **FRC Scout AI** — an AI-powered match prediction and scouting tool
for the 2026 FRC season.

Data is pulled live from [Statbotics](https://statbotics.io) and analyzed
locally by your Ollama model — no API keys or internet needed for AI.
""")

col1, col2 = st.columns(2)
with col1:
    st.info("**Match Analysis** — Predict win probabilities and scores for any 3v3 alliance matchup, with AI narrative analysis.")
with col2:
    st.info("**Team Breakdown** — Deep-dive into any team's 2026 EPA, auto/teleop/endgame splits, and rank.")

st.divider()
st.subheader("Setup")
st.markdown("""
1. Make sure [Ollama](https://ollama.com) is installed and running:
   ```bash
   ollama serve
   ```
2. Pull a model (if you haven't already):
   ```bash
   ollama pull llama3.2
   ```
3. Run the app from the `strategy/` directory:
   ```bash
   streamlit run scout/Home.py
   ```
4. Pick your model in the sidebar on any page.
""")

with st.expander("About the prediction model"):
    st.markdown("""
    The win probability model is a Random Forest trained on 2026 FRC match data,
    using EPA-based features from Statbotics (auto, teleop, endgame, win rate, etc.).
    Alliance scores are estimated as the sum of each team's EPA.

    Predictions are **EPA-driven** — scouting-specific features (driver skill,
    reliability, route choice) default to neutral values when no scouting CSV is loaded.
    """)
