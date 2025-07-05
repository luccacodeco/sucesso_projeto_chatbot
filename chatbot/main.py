import os
import streamlit as st
import pandas as pd
from agent import agent_executor  # Executor configurado no agent.py

# Função para carregar os dados dos usuários
@st.cache_data
def load_user_data():
    """
    Lê o CSV de usuários da pasta ml_model/data uma única vez e faz cache.
    """
    data_path = 'ml_model/data/usuarios.csv'
    return pd.read_csv(data_path)

# Configuração da página
st.set_page_config(
    page_title="LLM Chatbot de Previsão de Projetos",
    page_icon="🤖",
    layout="wide"
)

# CSS custom para deixar a sidebar preta semi-transparente
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título e separador
st.title("Chatbot de Previsão de Sucesso de Projetos")
st.markdown("---")

# Monta opções de usuários para selecionar na sidebar
users_df = load_user_data()
user_options = {
    f"{row['nome']} ({row['cargo']})": row for _, row in users_df.iterrows()
}

# Combobox de seleção de usuário
selected_user = st.sidebar.selectbox(
    "Selecione o usuário:",
    list(user_options.keys())
)

# Dados do usuário selecionado
user_info = user_options[selected_user]

# Exibe contexto do usuário na sidebar
st.sidebar.success(f"Nome: {user_info['nome']}")
st.sidebar.info(f"Cargo: {user_info['cargo']}")
st.sidebar.info(f"Projetos: {user_info['historico_projetos']}")
st.sidebar.info(f"Taxa de sucesso: {user_info['sucesso_medio']:.0%}")

# Garante que a sessão tem um histórico
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Mensagem inicial de boas-vindas
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Olá, sou o Chatbot de previsão de sucesso de projetos!\n\n"
            "Minhas funcionalidades incluem:\n"
            "- Fazer previsão de sucesso de um projeto com base em suas informações\n"
            "- Buscar o histórico dos projetos dos usuários\n\n"
            "Como posso te ajudar?"
        )
    })

# Exibe todo o histórico acumulado
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Campo de entrada do usuário
prompt = st.chat_input("Digite sua pergunta...")

if prompt:
    # Adiciona pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Monta contexto do usuário para passar ao AgentExecutor
    contexto_usuario = (
        f"O usuário é {user_info['nome']}, cargo {user_info['cargo']}, "
        f"com taxa de sucesso média de {user_info['sucesso_medio']:.0%}."
    )

    # Junta contexto + pergunta
    prompt_com_contexto = f"{contexto_usuario}\n\n{prompt}"

    # Executa o Agent com o input e mostra a resposta
    with st.chat_message("assistant"):
        resposta = agent_executor.invoke({
            "input": prompt_com_contexto
        })
        output = resposta["output"]
        st.markdown(output)

        # Guarda resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": output})
