## O deploy do chatbot e api foram feitos e está disponível para testar na url: https://chatbot-production-806b.up.railway.app/


# Treinamento do Modelo de Previsão de Sucesso de Projetos

Este projeto utiliza um modelo Random Forest para prever a probabilidade de sucesso de projetos com base em características estruturadas. A lógica de treinamento está definida no arquivo `train_model.py`.

## Descrição do Treinamento

O script `train_model.py` realiza as seguintes etapas principais:

1. **Carregamento dos Dados**
   - O script carrega um arquivo CSV (`ml_model/data/projetos.csv`) contendo dados históricos de projetos.
   - A coluna `data_inicio` é usada para derivar as variáveis de ano, mês e dia da semana, que são importantes para capturar sazonalidade e tendências temporais.

2. **Engenharia de Variáveis**
   - São definidas variáveis numéricas (exemplo: duração, orçamento, entregas, tamanho da equipe, recursos) e categóricas (tipo de projeto, departamento, complexidade, metodologia, risco).

3. **Divisão em Treino e Teste**
   - O conjunto de dados é dividido em 80% para treino e 20% para teste, usando estratificação para manter a proporção da variável alvo.

4. **Pipeline com Pré-Processamento**
   - É criado um `Pipeline` combinando escalonamento de variáveis numéricas (`StandardScaler`) e codificação one-hot para as variáveis categóricas (`OneHotEncoder`).
   - Isso garante que os dados estejam preparados corretamente antes de serem usados no Random Forest.

5. **Busca de Hiperparâmetros**
   - A busca por hiperparâmetros é feita com `RandomizedSearchCV`, usando validação cruzada estratificada repetida. Essa abordagem ajuda a encontrar uma combinação de parâmetros que maximize a área sob a curva ROC (AUC).

6. **Definição do Melhor Threshold**
   - Embora o Random Forest produza uma probabilidade, o script avalia diferentes thresholds de classificação para encontrar o ponto de corte que maximize a acurácia.
   - Esse threshold é salvo junto com o modelo para ser usado posteriormente em produção.

7. **Avaliação Final**
   - São calculadas métricas como Acurácia, ROC AUC, Precisão, Recall e F1 Score, além da matriz de confusão e um relatório de classificação.

8. **Persistência**
   - O pipeline final treinado e o threshold calculado são salvos em `ml_model/model.pkl` usando o `joblib` para serem carregados pela API de previsão.

## Por Que Essas Escolhas Foram Feitas

- **Random Forest** foi escolhido por ser robusto, lidar bem com variáveis categóricas codificadas e oferecer uma probabilidade de saída interpretável.
- **Pré-processamento com Pipeline** garante que as etapas de transformação sejam consistentes tanto em treino quanto em produção.
- **RandomizedSearchCV** foi usado porque permite uma busca eficiente em um espaço de parâmetros amplo sem custo computacional excessivo.
- **Threshold dinâmico** garante que o modelo seja mais ajustado ao contexto real de classificação (sucesso ou fracasso), evitando usar um corte padrão de 0.5 que pode não ser o ideal.

## Como Executar o Treinamento

1. **Tenha o ambiente configurado**
   - É necessário ter Python 3.9+ e todas as dependências listadas em `requirements.txt`. Para baixar, rode no terminal `pip install -r requirements.txt`

2. **Verifique o arquivo de dados**
   - O arquivo `ml_model/data/projetos.csv` deve estar presente.

3. **Execute o script**
   - No terminal, rode:

     ```
     python ml_model/train_model.py
     ```

   - O script irá carregar os dados, treinar o modelo, ajustar o threshold, mostrar métricas no terminal e salvar o arquivo `model.pkl` pronto para ser usado na API.

---

Este treinamento é o ponto de partida do pipeline de previsão de sucesso de projetos. Depois de treinado, o modelo é carregado pela API FastAPI para responder previsões em produção.


# API de Previsão de Sucesso de Projetos

Esta API, implementada com FastAPI, disponibiliza o modelo Random Forest treinado para prever o sucesso de projetos em produção. O código principal da API está no arquivo `api/main.py`.

## Descrição da Estrutura

A API foi projetada para receber informações detalhadas de um projeto, aplicar o pipeline treinado e retornar a probabilidade de sucesso junto com uma classificação final (sucesso ou fracasso) com base em um threshold otimizado.

## Principais Componentes

1. **Modelo e Threshold**
   - O modelo Random Forest e o threshold ajustado no treinamento são carregados a partir do arquivo `ml_model/model.pkl`.
   - Essa abordagem garante que as previsões usem o mesmo pipeline pré-processado usado no treinamento.

2. **Estrutura de Dados**
   - A entrada é validada usando o `Pydantic` para garantir que todos os campos obrigatórios estejam presentes no formato correto.
   - A resposta inclui a probabilidade de sucesso (float) e o resultado final (booleano).

3. **Eventos de Inicialização**
   - O carregamento do modelo é feito usando um `lifespan` do FastAPI. Assim, o pipeline é carregado uma única vez quando o servidor inicia, otimizando o tempo de resposta.

4. **Endpoints**
   - `GET /` — Verifica se a API está ativa.
   - `GET /health` — Verifica se o modelo foi carregado corretamente.
   - `POST /predict` — Recebe os dados do projeto, faz o pré-processamento embutido no pipeline, calcula a probabilidade e retorna o resultado.

## Por Que Essas Escolhas Foram Feitas

- **FastAPI** foi escolhido por sua rapidez de resposta e compatibilidade nativa com Pydantic para validação de dados.
- O pipeline salvo já inclui o pré-processamento, garantindo consistência entre treino e produção.
- O threshold calculado durante o treinamento garante que a classificação de sucesso seja baseada no melhor ponto de corte para o problema.

## Como Rodar em Produção

1. **Prepare o ambiente**
   - Verifique se o arquivo `ml_model/model.pkl` está presente e atualizado.
   - Garanta que as dependências do `requirements.txt` estejam instaladas.

2. **Execute a API com Uvicorn**
   - No terminal, rode:
     ```
     uvicorn api.main:app --host 0.0.0.0 --port 8000
     ```

   - Isso inicia o servidor FastAPI ouvindo em todas as interfaces na porta 8000.

## Testes

**Verifique a API**
- Acesse `http://localhost:8000/` para confirmar que está ativa.
- Acesse `http://localhost:8000/health` para verificar se o modelo foi carregado corretamente.

**Teste real com cURL**
- Envie uma requisição `POST` para `/predict` com um exemplo de projeto:
  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "duracao_meses": 12,
      "orcamento": 500000,
      "entregas": 3,
      "tamanho_equipe": 5,
      "recursos_disponiveis": 1,
      "ano_inicio": 2025,
      "mes_inicio": 7,
      "dia_semana": 2,
      "tipo_projeto": "Software",
      "departamento": "TI",
      "complexidade": "Média",
      "metodologia": "Scrum",
      "risco": "Baixo"
    }'

# Chatbot de Previsão de Sucesso de Projetos

Este módulo implementa um **Chatbot corporativo** usando **Streamlit** e **LangChain**, que interage com o usuário para prever o sucesso de projetos com base em um modelo Random Forest exposto por uma **API FastAPI**.

## Estrutura

O chatbot está dividido em **três partes principais**:

- `previsao.py`: Contém funções de normalização, parsing de texto, validação fuzzy match e chamada HTTP para a API FastAPI.
- `agent.py`: Define o executor de agente (`AgentExecutor`) usando LangChain, com prompt detalhado, regras de negócio e ferramentas para previsão de projetos e busca de histórico de usuários.
- `main.py`: Implementa a interface interativa usando **Streamlit**, carregando os dados de usuários e gerenciando o fluxo de perguntas e respostas.

## Por que essa estrutura

- **Separação clara:** A parte de pré-processamento (`previsao.py`) fica desacoplada do fluxo do LangChain.
- **Orquestração com LangChain:** O `agent.py` organiza o prompt com regras específicas, garantindo que o assistente siga as regras corporativas, interprete variações de texto, normalize entradas e use as ferramentas corretas.
- **Interface amigável:** O `main.py` roda um **chat em tempo real** com histórico persistente, exibindo contexto do usuário e personalizando as respostas.

## Funcionalidades

- Recebe dados de um projeto, valida campos, normaliza entradas inconsistentes (como “8 meses”, “1 milhão”, “baixo”, etc).
- Faz fuzzy match para campos categóricos como tipo de projeto, departamento, complexidade, metodologia e risco.
- Consulta histórico de usuários a partir de um CSV.
- Invoca a API FastAPI para calcular a probabilidade de sucesso.
- Gera uma recomendação curta e corporativa usando OpenAI.
- Mostra o histórico completo do diálogo com o usuário em tempo real.

## Requisitos

- Python 3.10 ou superior.
- Streamlit.
- LangChain e LangChain OpenAI.
- Biblioteca `dotenv` para variáveis de ambiente.
- API FastAPI em execução para receber as requisições de previsão.

## Como Executar Localmente

Para rodar o chatbot **localmente**, é necessário garantir que **a API FastAPI esteja ativa** em paralelo, pois ela faz o cálculo da probabilidade de sucesso e com o requirements.txt instalado.

### Passo a Passo

1. **Inicie a API**
   Execute no terminal, dentro da pasta da API:
    uvicorn main:app --reload --port 8000


2. **Configure o arquivo `.env`**

Crie um arquivo `.env` dentro da pasta `chatbot/` ou na raiz do projeto com o seguinte conteúdo:

API_BASE_URL=http://localhost:8000
OPENAI_API_KEY= chave da OpenAI

4. **Rode o chatbot**
No terminal execute:
streamlit run chatbot/main.py


5. **Acesse no navegador**
O Streamlit mostrará o link (geralmente `http://localhost:8501`).

6. **Pronto**
O chatbot estará funcional para receber perguntas, gerar previsões de projetos e consultar o histórico de usuários.

## Observações

- Toda a comunicação entre chatbot e modelo é feita via HTTP (API REST).
- As respostas são ajustadas em português corporativo.
- A variável `OPENAI_API_KEY` é obrigatória para gerar as recomendações do assistente com LangChain.
- Se alterar a porta da API, ajuste também o `API_BASE_URL` no `.env`.

---


