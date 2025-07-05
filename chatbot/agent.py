from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
import pandas as pd
import os
from previsao import (
    predict_project_success,
    format_prediction_response,
    get_missing_fields,
    normalize_project_data
)

# Função principal de previsão
def prever_projeto_tool(
    duracao_meses: int,
    orcamento: float,
    entregas: int,
    tamanho_equipe: int,
    recursos_disponiveis: str,
    data_inicio: str,
    tipo_projeto: str,
    departamento: str,
    complexidade: str,
    metodologia: str,
    risco: str,
) -> str:
    """
    Recebe os dados do projeto, normaliza, valida, faz a previsão
    e gera uma recomendação resumida.
    """
    project_data = {
        'duracao_meses': duracao_meses,
        'orcamento': orcamento,
        'entregas': entregas,
        'tamanho_equipe': tamanho_equipe,
        'recursos_disponiveis': recursos_disponiveis,
        'data_inicio': data_inicio,
        'tipo_projeto': tipo_projeto,
        'departamento': departamento,
        'complexidade': complexidade,
        'metodologia': metodologia,
        'risco': risco
    }

    # Normaliza valores e faz parsing de texto
    project_data = normalize_project_data(project_data)

    # Verifica campos obrigatórios faltantes
    missing = get_missing_fields(project_data)
    if missing:
        return f"Atenção: Campos obrigatórios ausentes ou inválidos: {missing}"

    # Chama a previsão na API
    prediction = predict_project_success(project_data)
    base_result = format_prediction_response(prediction, project_data)

    # Pede ao modelo uma recomendação curta e corporativa
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    user_prompt = f"""
    O resultado da previsão é:

    {base_result}

    Gere uma recomendação curta, corporativa, clara, em 2 linhas no máximo,
    levando em consideração os dados do projeto, como duração, orçamento,
    recursos, tipo, complexidade, metodologia e risco.
    """

    recommendation = llm.invoke(user_prompt).content.strip()

    return f"{base_result}\n\nRecomendação:\n{recommendation}"


# Registra a função como StructuredTool
previsao_tool = StructuredTool.from_function(
    prever_projeto_tool,
    name="PreverProjeto",
    description="Prevê o sucesso de um projeto."
)

# Função para histórico de usuário

# Carrega o CSV de usuários uma única vez
DATA_PATH = 'ml_model/data/usuarios.csv'
USERS_DF = pd.read_csv(DATA_PATH)

def buscar_historico_usuario(nome: str) -> dict:
    """
    Busca o histórico do usuário pelo nome exato (sem case sensitive).
    """
    nome = nome.strip().lower()
    df_match = USERS_DF[USERS_DF['nome'].str.lower().str.strip() == nome]

    if df_match.empty:
        return {"found": False, "nome": nome}

    row = df_match.iloc[0]
    return {
        "found": True,
        "nome": row['nome'],
        "cargo": row['cargo'],
        "historico_projetos": int(row['historico_projetos']),
        "experiencia_anos": int(row['experiencia_anos']),
        "sucesso_medio": float(row['sucesso_medio'])
    }

# Registra a função como StructuredTool
historico_tool = StructuredTool.from_function(
    buscar_historico_usuario,
    name="BuscarHistoricoUsuario",
    description="Busca o histórico de um usuário pelo nome."
)

# Prompt principal com instruções
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a corporate assistant specialized in predicting the success of projects. "
     "Whenever the user says they want to make a prediction, you must explain that you need the following 11 pieces of information, "
     "always in this exact order:\n"
     "1. Project duration (in months).\n"
     "2. Project budget (e.g., 1 million, 200 thousand, 100000).\n"
     "3. Number of planned deliveries.\n"
     "4. Team size (number of people).\n"
     "5. Available resources: Low, Medium, or High.\n"
     "6. Project start date (e.g., 20/07/2025).\n"
     "7. Type of project: Software, Infrastructure, Research, or Construction.\n"
     "8. Responsible department: IT, Marketing, HR, Operations, Finance.\n"
     "9. Project complexity: Low, Medium, or High.\n"
     "10. Methodology: Agile, Waterfall, Scrum, Kanban, or XP.\n"
     "11. Risk level: Low, Medium, or High.\n\n"

     "Important rule: If all 11 values are present in the user's reply, use them immediately, "
     "even if they contain extra words such as 'entregas', 'entrega', 'deliveries', 'delivery', 'months', 'people', or 'team'. "
     "Internally clean the value and extract only the number for each numeric field. "
     "Never ask for the full list again if you already have all 11 values.\n\n"

     "Special rule for Available Resources:\n"
     "Always accept variations for Available Resources:\n"
     "‘Baixo’, ‘Baixa’, ‘Baixos’, ‘Baixas’ mean Low (0).\n"
     "‘Médio’, ‘Medio’, ‘Médios’, ‘Medios’, ‘Média’, ‘Media’, ‘Médias’, ‘Medias’ mean Medium (1).\n"
     "‘Alto’, ‘Alta’, ‘Altos’, ‘Altas’ mean High (2).\n"
     "Accept them exactly as the user writes them, in any uppercase or lowercase, with or without accent. "
     "Internally, always convert Low to 0, Medium to 1, High to 2. "
     "Never ask for confirmation if the word matches any of these variations.\n\n"

     "If the user says any number with words like 'entregas', 'entrega', 'deliveries', 'delivery', 'meses', 'months', 'people', 'pessoas', or 'team', "
     "treat it as just the number immediately. Do not ask for confirmation if the number is clear.\n\n"

     "If any value is missing or the number is unclear (for example: 'some deliveries'), then and only then ask "
     "for confirmation of the specific field in a clear and corporate tone. "
     "Example: 'Sorry, could you please confirm the exact number of deliveries?'. "
     "If the number is clear, like '3 deliveries', do not ask for confirmation.\n\n"

     "Never confuse the Available Resources field (Low, Medium, High) with the Risk Level field (Low, Medium, High). "
     "These are distinct fields and should only be confirmed separately if unclear.\n\n"

     "Always ask for and confirm information clearly, politely, and in corporate Brazilian Portuguese. "
     "Whenever any field is missing, ask using the same standard, keeping the exact order of the 11 items. "
     "When providing the prediction result, do not show the raw data the user provided. "
     "Use the user context (name, role, or success rate) to briefly explain in one short sentence "
     "how their experience or history may influence the project's success.\n\n"

     "If the user asks about someone’s history, use the history tool and write a short line praising or suggesting improvements. "
     "If the user does not specify a name, ask whose history they would like to know.\n\n"

     "If the user asks anything outside the scope of project predictions or user history, "
     "explain that you are a corporate chatbot focused on project success predictions and user history only. "
     "If the user asks what you do, answer that you are a chatbot for project success prediction "
     "and that you can also provide user history.\n\n"

     "When replying in Portuguese, always keep the official methodology names exactly as they are: Agile, Waterfall, Scrum, Kanban, XP. Never translate them."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("human", "{agent_scratchpad}")
])



# Memória do chat
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Executor do Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

agent = create_openai_functions_agent(
    llm,
    [previsao_tool, historico_tool],
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[previsao_tool, historico_tool],
    memory=memory,
    verbose=True
)
