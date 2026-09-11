import streamlit as st
import torch
import re

from transformers import (
    BertForSequenceClassification,
    BertConfig,
    BertTokenizer
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "best_bert_model.pt"
CONFIG_PATH = "config.json"
TOKENIZER_PATH = "."

MAX_LEN = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# Text Preprocessing
# ============================================================

def clean_text_for_bert(text):

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove Twitter usernames
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# Load Tokenizer
# ============================================================

@st.cache_resource
def load_tokenizer():

    tokenizer = BertTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True
    )

    return tokenizer


# ============================================================
# Load Fine-Tuned BERT Model
# ============================================================

@st.cache_resource
def load_model():

    # --------------------------------------------------------
    # Load configuration from local config.json
    # --------------------------------------------------------

    config = BertConfig.from_pretrained(
        CONFIG_PATH,
        local_files_only=True
    )

    # --------------------------------------------------------
    # Create BERT architecture from configuration
    # This does NOT download bert-base-uncased
    # --------------------------------------------------------

    model = BertForSequenceClassification(config)

    # --------------------------------------------------------
    # Load your fine-tuned weights
    # --------------------------------------------------------

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # Move model to CPU/GPU
    # --------------------------------------------------------

    model.to(DEVICE)

    # Evaluation mode
    model.eval()

    return model


# ============================================================
# Load Model + Tokenizer
# ============================================================

try:

    tokenizer = load_tokenizer()
    model = load_model()

    model_status = True

except Exception as e:

    model_status = False

    st.error(
        f"Model loading failed:\n\n{e}"
    )


# ============================================================
# Prediction Function
# ============================================================

def predict_sentiment(text):

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    cleaned_text = clean_text_for_bert(text)

    # --------------------------------------------------------
    # BERT Tokenization
    # --------------------------------------------------------

    encoding = tokenizer(
        cleaned_text,

        add_special_tokens=True,

        max_length=MAX_LEN,

        padding="max_length",

        truncation=True,

        return_attention_mask=True,

        return_tensors="pt"
    )

    # --------------------------------------------------------
    # Move tensors to device
    # --------------------------------------------------------

    input_ids = encoding[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    # --------------------------------------------------------
    # Convert class to sentiment
    # --------------------------------------------------------

    if predicted_class == 0:

        sentiment = "Negative"

    else:

        sentiment = "Positive"

    return sentiment, confidence


# ============================================================
# Streamlit Interface
# ============================================================

st.title("🤖 BERT Sentiment Analysis")

st.write(
    "Enter a sentence or tweet and the fine-tuned "
    "BERT model will predict its sentiment."
)

st.divider()


text = st.text_area(
    "Enter your text:",
    placeholder="Example: I absolutely love this product!",
    height=150
)


# ============================================================
# Prediction Button
# ============================================================

if st.button(
    "Predict Sentiment",
    type="primary"
):

    if not model_status:

        st.error(
            "Model could not be loaded."
        )

    elif not text.strip():

        st.warning(
            "Please enter some text."
        )

    else:

        sentiment, confidence = predict_sentiment(
            text
        )

        st.subheader("Prediction")

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        if sentiment == "Positive":

            st.success(
                f"😊 {sentiment}"
            )

        else:

            st.error(
                f"😞 {sentiment}"
            )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        st.write(
            f"Confidence: **{confidence * 100:.2f}%**"
        )

        st.progress(
            confidence
        )

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        st.divider()

        st.write("**Input:**")

        st.write(text)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Fine-tuned BERT Sentiment Classification Model"
)
