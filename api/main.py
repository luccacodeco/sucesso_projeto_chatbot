from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from contextlib import asynccontextmanager
import uvicorn

# Variáveis globais
model = None
threshold = None  # Threshold de corte para classificar sucesso ou fracasso

# Esquema de entrada da API
class ProjetoRequest(BaseModel):
    """
    Estrutura de dados que define os campos obrigatórios
    que o cliente deve enviar para realizar a previsão.
    """
    duracao_meses: int
    orcamento: float
    entregas: int
    tamanho_equipe: int
    recursos_disponiveis: int
    ano_inicio: int
    mes_inicio: int
    dia_semana: int
    tipo_projeto: str
    departamento: str
    complexidade: str
    metodologia: str
    risco: str

# Esquema de resposta da API
class ProjetoResponse(BaseModel):
    """
    Estrutura de resposta que será retornada.
    Inclui a probabilidade de sucesso e a classificação final.
    """
    probabilidade_sucesso: float
    sucesso: bool

# Função de carregamento do modelo
def load_model():
    """
    Carrega o pipeline Random Forest treinado e o threshold salvo
    do arquivo .pkl
    """
    global model, threshold
    bundle = joblib.load('ml_model/model.pkl')
    model = bundle['model']
    threshold = bundle['threshold']
    print(f"Modelo carregado. Threshold: {threshold:.2f}")

# Lifespan Event para inicialização
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Evento de ciclo de vida do FastAPI para rodar rotinas
    na inicialização. Aqui, carrega o modelo e o threshold.
    """
    load_model()
    yield

# Criação da aplicação FastAPI
app = FastAPI(
    title="API de Previsão de Sucesso de Projetos",
    description="API para prever o sucesso de projetos usando Random Forest treinado.",
    version="1.0.0",
    lifespan=lifespan
)

# Rota principal de teste rápido
@app.get("/")
async def root():
    """
    Endpoint principal apenas para teste de disponibilidade da API.
    """
    return {"message": "API de Previsão de Sucesso de Projetos está ativa."}

# Rota de health check
@app.get("/health")
async def health_check():
    """
    Endpoint para verificar se o modelo foi carregado corretamente.
    """
    return {"status": "healthy", "model_loaded": model is not None}

# Rota principal de previsão
@app.post("/predict", response_model=ProjetoResponse)
async def predict(projeto: ProjetoRequest):
    """
    Recebe os dados do projeto, faz a previsão da probabilidade de sucesso
    e aplica o threshold salvo para classificar como sucesso ou fracasso.
    """
    if model is None or threshold is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    # Monta DataFrame com as features de entrada
    df = pd.DataFrame([projeto.model_dump()])

    # Calcula probabilidade de sucesso (classe positiva)
    proba = model.predict_proba(df)[0][1]

    # Classifica como sucesso ou fracasso com base no threshold
    sucesso = proba >= threshold

    return ProjetoResponse(
        probabilidade_sucesso=round(float(proba), 4),
        sucesso=sucesso
    )


