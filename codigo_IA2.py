import streamlit as st
from transformers import pipeline

@st.cache_resource
def carregar_modelo():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = carregar_modelo()

st.title("Assistente Jurídico Grátis")

texto_base = """
O direito penal é o ramo do direito que trata dos crimes e das penas aplicadas aos infratores.
Por exemplo, o homicídio simples no Brasil tem pena de reclusão de 6 a 20 anos, conforme o Código Penal.
O direito civil regula relações entre pessoas físicas e jurídicas, como contratos, propriedade, família.
Um contrato é um acordo que cria obrigações jurídicas entre partes.
A Constituição Federal de 1988 é a lei máxima no Brasil e rege os direitos fundamentais.
Prescrição é o tempo máximo para reclamar um direito na justiça, que varia conforme o tipo de ação.
"""

st.write("### Contexto Jurídico Base:")
st.write("Faça a sua pergunta:")

pergunta = st.text_input("Digite sua dúvida jurídica:")

if st.button("Responder"):
    if pergunta.strip():
        resultado = qa_pipeline(question=pergunta, context=texto_base)
        st.write("### Resposta:")
        st.write(resultado['answer'])
    else:
        st.warning("Por favor, digite uma pergunta.")
