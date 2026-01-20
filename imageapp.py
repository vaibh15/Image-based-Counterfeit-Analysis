# ==================================================
# PRAMAAN – Image-based Counterfeit Risk Analyzer
# (Deterministic, Safe, Single-file Streamlit App)
# ==================================================

import os
import json
import base64
import requests
import streamlit as st
#from dotenv import load_dotenv
from io import BytesIO
from PIL import Image

from openai import OpenAI, BadRequestError

# ==================================================
# 1. ENV SETUP
# ==================================================
#load_dotenv()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-4.1-mini"  # Vision-capable

ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"]

# ==================================================
# 2. DECISION LOGIC (EXPLICIT & EXPLAINABLE)
# ==================================================
def decide_suspected_counterfeit(risk_percent):
    if risk_percent is None:
        return None
    return risk_percent >= 60


def decide_confidence_level(risk_percent):
    if risk_percent is None:
        return "low"
    if risk_percent >= 80:
        return "high"
    elif risk_percent >= 50:
        return "medium"
    else:
        return "low"

# ==================================================
# 3. HELPERS
# ==================================================
def image_to_base64(uploaded_file) -> str:
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        return base64.b64encode(uploaded_file.read()).decode("utf-8")
    except Exception:
        raise ValueError("Uploaded file is not a valid image.")


def download_image_as_base64(url: str) -> str:
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
    except Exception:
        raise ValueError("Failed to access the image URL.")

    if response.status_code != 200:
        raise ValueError("Image URL could not be accessed.")

    content_type = response.headers.get("Content-Type", "").lower()
    if not any(t in content_type for t in ALLOWED_IMAGE_TYPES):
        raise ValueError(
            "URL does not point to a valid image (PNG, JPG, or WEBP only)."
        )

    try:
        img = Image.open(BytesIO(response.content))
        img.verify()
    except Exception:
        raise ValueError("Downloaded file is not a valid image.")

    return base64.b64encode(response.content).decode("utf-8")


def analyze_image_with_gpt(image_base64: str) -> dict:
    system_prompt = """
You are a product authenticity and counterfeit risk analyst.

You analyze a single product image to estimate the likelihood that the product
shown is counterfeit, replica, or misleadingly represented as original.

Rules:
- Base conclusions ONLY on visible evidence.
- Consider logo accuracy, typography, materials, finishing,
  packaging quality, spelling errors, serial markings, and design consistency.
- Be conservative when evidence is weak or image quality is poor.
- Explicitly state uncertainty when applicable.
- Do NOT decide final verdicts or confidence thresholds.

Respond ONLY in strict JSON.

JSON format:
{
  "counterfeit_risk_percent": number,
  "reasoning": string,
  "visual_red_flags": string[],
  "visual_green_flags": string[]
}
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the following product image for counterfeit risk."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    )

    try:
        return json.loads(completion.choices[0].message.content)
    except Exception:
        return {
            "counterfeit_risk_percent": None,
            "reasoning": "Failed to parse model output.",
            "visual_red_flags": [],
            "visual_green_flags": []
        }

# ==================================================
# 4. STREAMLIT UI
# ==================================================
st.set_page_config(
    page_title="PRAMAAN – Image Counterfeit AnalyzerPRAMAAN – AI-Based Counterfeit Risk Analysis for Product Images",
    layout="centered"
)

st.markdown("""
## **PRAMAAN – Image-based Counterfeit Risk Analyzer**

Upload a product image or provide a **direct image URL**.
The system estimates counterfeit risk based on **visual indicators only**.
""")

# -----------------------------
# IMAGE INPUT
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload product image",
    type=["jpg", "jpeg", "png", "webp"]
)

image_url = st.text_input(
    "OR paste image URL",
    placeholder="https://example.com/product.jpg"
)

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("Analyze Image"):

    if not uploaded_file and not image_url:
        st.error("Please upload an image or provide an image URL.")
        st.stop()

    # -----------------------------
    # LOAD + VALIDATE IMAGE
    # -----------------------------
    try:
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            image_b64 = image_to_base64(uploaded_file)
        else:
            st.image(image_url, caption="Image from URL", use_container_width=True)
            image_b64 = download_image_as_base64(image_url)

    except ValueError as ve:
        st.error(str(ve))
        st.stop()

    except Exception:
        st.error("Failed to load image. Please provide a valid product image.")
        st.stop()

    # -----------------------------
    # OPENAI ANALYSIS
    # -----------------------------
    with st.spinner("Analyzing image for counterfeit risk..."):
        try:
            analysis = analyze_image_with_gpt(image_b64)
        except BadRequestError:
            st.error(
                "The image could not be processed. "
                "Please try a clear JPG, PNG, or WEBP image."
            )
            st.stop()

    # -----------------------------
    # APPLY DETERMINISTIC RULES
    # -----------------------------
    risk = analysis.get("counterfeit_risk_percent")

    suspected = decide_suspected_counterfeit(risk)
    confidence = decide_confidence_level(risk)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.subheader("🔍 Counterfeit Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk %", f"{risk}%" if risk is not None else "N/A")

    with col2:
        st.metric(
            "Suspected Counterfeit",
            "Yes" if suspected else "No"
        )

    with col3:
        st.metric(
            "Confidence",
            confidence.capitalize()
        )

    # Risk band explanation
    if risk is not None:
        if risk >= 80:
            st.error("High counterfeit risk based on strong visual indicators.")
        elif risk >= 50:
            st.warning("Moderate counterfeit risk with some suspicious indicators.")
        else:
            st.success("Low counterfeit risk based on available visual evidence.")

    st.subheader("🧠 Reasoning")
    st.write(analysis.get("reasoning", ""))

    st.subheader("🚩 Visual Red Flags")
    if analysis.get("visual_red_flags"):
        for rf in analysis["visual_red_flags"]:
            st.write(f"- {rf}")
    else:
        st.write("None detected")

    st.subheader("✅ Visual Green Flags")
    if analysis.get("visual_green_flags"):
        for gf in analysis["visual_green_flags"]:
            st.write(f"- {gf}")
    else:
        st.write("None detected")

    with st.expander("🧠 Raw AI Output (JSON)"):
        st.json({
            **analysis,
            "suspected_is_counterfeit": suspected,
            "confidence_level": confidence
        })
