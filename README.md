# 🤖 LLM Chatbot de Previsão de Sucesso de Projetos

Sistema completo de machine learning para prever o sucesso de novos projetos, incluindo modelo de ML tradicional, API REST e chatbot interativo baseado em Large Language Model (LLM) com interface web.

## 📋 Visão Geral

Este projeto implementa um sistema completo de previsão de sucesso de projetos que combina:

1. **Modelo de ML Tradicional**: Regressão logística treinada com dados históricos
2. **API REST**: Deploy simplificado do modelo com FastAPI
3. **LLM Chatbot Interativo**: Interface web com Streamlit e OpenAI GPT para interação inteligente com usuários

## 🏗️ Arquitetura do Sistema

```
chatbot_model/
├── ml_model/              # Modelo de Machine Learning
│   ├── data/
│   │   ├── projetos.csv   # Dados históricos de projetos
│   │   └── usuarios.csv   # Base de dados de usuários
│   ├── train_model.py     # Script de treinamento
│   ├── model.pkl          # Modelo treinado (gerado)
│   ├── scaler.pkl         # Scaler para normalização (gerado)
│   └── README.md          # Documentação do modelo
│
├── api/                   # API REST
│   ├── main.py           # Aplicação FastAPI
│   └── README.md         # Documentação da API
│
├── chatbot/              # Interface Web
│   ├── chatbot.py        # Aplicação Streamlit
│   └── README.md         # Documentação do chatbot
│
├── requirements.txt      # Dependências do projeto
└── README.md            # Este arquivo
```

## 🚀 Instalação e Configuração

### 1. Pré-requisitos
- Python 3.8+
- pip (gerenciador de pacotes Python)

### 2. Instalar Dependências
```bash
# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 3. Configurar LLM (Opcional)
Para usar a funcionalidade de LLM:

1. **Criar arquivo .env**:
```bash
cp env.example .env
```

2. **Adicionar sua API key da OpenAI**:
```bash
echo "OPENAI_API_KEY=sua_chave_aqui" >> .env
```

3. **Obter API key da OpenAI**:
- Acesse: https://platform.openai.com/api-keys
- Crie uma nova chave
- Copie para o arquivo .env

### 4. Treinar o Modelo
```bash
cd ml_model
python train_model.py
```

### 5. Executar a API
```bash
cd api
python main.py
```

### 6. Executar o Chatbot
```bash
cd chatbot
streamlit run chatbot.py
```

## 🎯 Como Usar

### 1. Acessar o Sistema
- **API**: http://localhost:8000
- **Documentação da API**: http://localhost:8000/docs
- **Chatbot**: http://localhost:8501

### 2. Interagir com o Chatbot
1. **Selecionar usuário** na barra lateral
2. **Fazer perguntas** no chat ou usar o formulário rápido
3. **Receber previsões** e recomendações personalizadas

### 3. Exemplo de Uso
```
Usuário: "Tenho um projeto que vai durar 6 meses, orçamento de R$ 500.000, equipe de 8 pessoas, recursos médios. O que você acha?"

Chatbot: 🎯 Analisando seu projeto...

Com base nos dados fornecidos e considerando sua experiência como Gerente de TI com 80% de taxa de sucesso histórica, posso fazer uma análise detalhada:

📊 **Previsão de Sucesso:**
- Probabilidade: 72.5%
- Resultado: ✅ SUCESSO

💡 **Recomendações Personalizadas:**
• Com sua experiência de 15 projetos anteriores, você tem boa base para gerenciar este projeto
• Considere aumentar o orçamento para R$ 600-700k para maior margem de segurança
• Mantenha a equipe de 8 pessoas, mas garanta que todos tenham experiência relevante
```

## 📊 Dados e Modelo

### Features do Modelo
- **Duração do projeto** (meses)
- **Orçamento** (R$)
- **Tamanho da equipe** (número de pessoas)
- **Recursos disponíveis** (Baixo/Médio/Alto)

### Base de Dados
- **30 projetos históricos** para treinamento
- **10 usuários** com perfis profissionais
- **Dados fictícios** realistas para demonstração

### Performance Esperada
- **Acurácia**: ~70-80%
- **Precisão**: ~75-85%
- **Recall**: ~70-80%

## 🔧 Configuração Avançada

### Variáveis de Ambiente
```bash
# API
export API_HOST=0.0.0.0
export API_PORT=8000

# Chatbot
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Personalização
1. **Dados**: Substitua `projetos.csv` e `usuarios.csv` com seus dados
2. **Modelo**: Modifique `train_model.py` para usar outros algoritmos
3. **Recomendações**: Ajuste a função `generate_recommendations()` na API
4. **Interface**: Personalize o chatbot em `chatbot.py`

## 📈 Funcionalidades

### Modelo de ML
- ✅ Treinamento com regressão logística
- ✅ Normalização de features
- ✅ Avaliação com múltiplas métricas
- ✅ Serialização do modelo

### API REST
- ✅ Endpoint de previsão
- ✅ Validação de dados
- ✅ Geração de recomendações
- ✅ Documentação automática (Swagger)

### LLM Chatbot
- ✅ Interface conversacional
- ✅ Integração com OpenAI GPT
- ✅ Processamento de linguagem natural avançado
- ✅ Seleção de usuário
- ✅ Formulário rápido
- ✅ Análise personalizada com contexto
- ✅ Respostas inteligentes e contextualizadas

## 🛠️ Tecnologias Utilizadas

### Backend
- **Python**: Linguagem principal
- **scikit-learn**: Machine learning
- **FastAPI**: Framework web para API
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **OpenAI**: LLM para processamento de linguagem natural

### Frontend
- **Streamlit**: Interface web
- **HTML/CSS**: Estilização
- **JavaScript**: Interatividade

### Infraestrutura
- **joblib**: Serialização de modelos
- **requests**: Comunicação HTTP
- **uvicorn**: Servidor ASGI

## 🔒 Segurança

- **Validação de entrada**: Pydantic para API
- **Sanitização**: Limpeza de dados
- **Tratamento de erros**: Mensagens amigáveis
- **Timeout**: Controle de tempo de resposta

## 📊 Monitoramento

### Logs
- Treinamento do modelo
- Requisições da API
- Interações do chatbot

### Métricas
- Performance do modelo
- Uso da API
- Satisfação do usuário

## 🚀 Deploy

### Desenvolvimento
```bash
# Terminal 1 - API
cd api && python main.py

# Terminal 2 - Chatbot
cd chatbot && streamlit run chatbot.py
```

### Produção
1. **Configurar servidor web** (nginx)
2. **Usar Gunicorn** para API
3. **Configurar SSL**
4. **Implementar autenticação**
5. **Configurar monitoramento**

## 🎯 Próximos Passos

### Melhorias Sugeridas
1. **NLP avançado**: Integração com spaCy
2. **Mais algoritmos**: Random Forest, XGBoost
3. **Análise temporal**: Tendências ao longo do tempo
4. **Dashboard**: Métricas e visualizações
5. **Integração**: APIs externas de projetos

### Funcionalidades Futuras
- **Análise de sentimentos**
- **Recomendações automáticas**
- **Relatórios em PDF**
- **Integração com calendário**
- **Notificações por email**

## 📝 Licença

Este projeto é de código aberto e pode ser usado para fins educacionais e comerciais.

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique a documentação em cada pasta
2. Consulte os logs de erro
3. Abra uma issue no repositório

---

**Desenvolvido com ❤️ para demonstrar o poder do Machine Learning em projetos reais.** 