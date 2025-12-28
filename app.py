import streamlit as st
import tempfile
import pandas as pd
from model.predict import predict_top_k

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Bird Species Identifier",
    page_icon="🦜",
    layout="centered"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
.block-container {
    max-width: 900px;
    padding-top: 2rem;
}
.pred-box {
    padding: 1rem;
    border-radius: 10px;
    background-color: #1f2937;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.title("🦜 Bird Species Identifier")
st.caption("Upload a bird image and let AI identify the species")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a bird image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # ----------------------------
    # RUN PREDICTION
    # ----------------------------
    with st.spinner("🧠 Analyzing image..."):
        results = predict_top_k(image_path, k=5)

    best = results[0]

    # ----------------------------
    # CONFIDENCE MESSAGE
    # ----------------------------
    if best["confidence"] < 0.3:
        st.warning(
            "⚠️ Low confidence prediction. "
            "This bird may look similar to multiple species."
        )
    else:
        st.success("✅ Model is confident in this prediction.")

    # ----------------------------
    # MAIN RESULT
    # ----------------------------
    st.markdown(
        f"""
        <div class="pred-box">
        <h3>🏆 Predicted Species</h3>
        <h2>{best["species"]}</h2>
        <p>Confidence: <b>{best["confidence"]*100:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------
    # TOP-K TABLE
    # ----------------------------
    st.subheader("📊 Confidence Breakdown")

    for row in results:
        st.write(f"**{row['species']}**")
        st.progress(min(row["confidence"], 1.0))

    df = pd.DataFrame(results)
    df["Confidence (%)"] = df["confidence"] * 100
    df = df.drop(columns=["confidence"])

    st.dataframe(
    df,
    use_container_width=True
    )

    # ----------------------------
    # MODEL INFO
    # ----------------------------
    with st.expander("🧠 Model Information"):
        st.markdown("""
        **Architecture:** EfficientNet  
        **Input Size:** 300 × 300 RGB  
        **Training:** Transfer Learning + Fine-Tuning  
        **Output:** Softmax probabilities over bird species  

        **Limitations:**
        - Similar species may confuse the model
        - Lighting and angle affect accuracy
        """)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow & Streamlit")