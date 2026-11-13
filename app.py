import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

BASE_MODEL = "QCRI/Fanar-1-9B-Instruct"
ADAPTER = "mwael399/CarHub-Fanar-1-9B-QLoRA"

st.set_page_config(
    page_title="CarHub AI Assistant",
    page_icon="🚗"
)

st.title("🚗 CarHub AI Assistant")
st.caption("AI-powered Automotive Maintenance Assistant")

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        model,
        ADAPTER
    )

    model.eval()

    return tokenizer, model


with st.spinner("Loading CarHub AI..."):
    tokenizer, model = load_model()

question = st.chat_input("Describe your car problem...")

if question:

    st.chat_message("user").write(question)

    messages = [
        {
            "role": "system",
            "content": (
                "You are CarHub, an automotive maintenance assistant. "
                "Answer in Egyptian Arabic. Explain possible causes, "
                "recommended checks, and safety warnings. "
                "Never claim a diagnosis is certain without evidence."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with st.spinner("CarHub is thinking..."):

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

    st.chat_message("assistant").write(answer)
