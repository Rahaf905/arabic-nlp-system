import streamlit as st
import joblib
import torch
import re
import emoji
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


# Page Config


st.set_page_config(
    page_title="Arabic NLP System",
    layout="centered"
)

st.title("Arabic NLP Question Answering System")

# Preprocessing Functions


def arabic_preprocessing(text):
    text = str(text)

    text = emoji.replace_emoji(text, replace=" ")

    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ء", "", text)

    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    text = re.sub(r"ـ", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def classification_preprocessing(text):
    text = arabic_preprocessing(text)

    text = re.sub(r"[^ء-ي\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load Models

@st.cache_resource
def load_models():

    qa_model_path = "best_qa_model/best_qa_model"

    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_path,use_fast=False)

    qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_path)

    classifier = joblib.load("qwen_svm_classifier.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    qwen_model_name = "Qwen/Qwen3-Embedding-0.6B"

    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name,trust_remote_code=True)

    qwen_model = AutoModel.from_pretrained(qwen_model_name,trust_remote_code=True)

    qa_model.eval()
    qwen_model.eval()

    return (
        qa_tokenizer,
        qa_model,
        classifier,
        label_encoder,
        qwen_tokenizer,
        qwen_model
    )


qa_tokenizer, qa_model, classifier, label_encoder, qwen_tokenizer, qwen_model = load_models()

# Translation API

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY"
)


def generate_answer(question):

    input_text = "أجب عن السؤال التالي: " + question

    inputs = qa_tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = qa_model.generate(
            **inputs,
        max_new_tokens=80,
        num_beams=5
        )

    answer = qa_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# Qwen Embedding Function


def get_qwen_embedding(text):

    inputs = qwen_tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = qwen_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.float().cpu().numpy()

def classify_question(question):
    clean_question = classification_preprocessing(question)
    question_embedding = get_qwen_embedding(clean_question)
    pred_encoded = classifier.predict(question_embedding)[0]
    category = label_encoder.inverse_transform([pred_encoded])[0]
    return category


def translate_output(question, answer, category):

    text = f"""
Original Question:
{question}

Generated Answer:
{answer}

Predicted Category:
{category}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional Arabic to English translator. Translate accurately and clearly."
            },
            {
                "role": "user",
                "content": f"Translate the following content to English only:\n\n{text}"
            }
        ],
        max_tokens=500,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()




question = st.text_input("Enter Arabic Question:")

if st.button("Generate Result"):

    if question.strip() == "":
        st.warning("Please enter an Arabic question.")

    else:
        with st.spinner("Generating Arabic answer using fine-tuned T5..."):
            answer = generate_answer(question)

        with st.spinner("Predicting category using Qwen embeddings + SVM..."):
            category = classify_question(question)

        with st.spinner("Translating output using API..."):
            translated_output = translate_output(
                question,
                answer,
                category
            )

        st.subheader("Original Question")
        st.write(question)

        st.subheader("Generated Arabic Answer")
        st.write(answer)

        st.subheader("Predicted Category")
        st.write(category)

        st.subheader("Translated Output")
        st.write(translated_output)