import os
import json
import re
import difflib
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import requests

# Carrega variáveis de ambiente (como URL da API)
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

# Lista de campos categóricos que terão validação por fuzzy match
CATEGORICAL_FIELDS = [
    ('tipo_projeto', 'tipo_projeto'),
    ('departamento', 'departamento'),
    ('complexidade', 'complexidade'),
    ('metodologia', 'metodologia'),
    ('risco', 'risco'),
]

# Cache interno para não recarregar valores válidos toda hora
_categorical_values_cache = None

def get_categorical_field_values():
    """
    Lê os valores únicos de cada campo categórico a partir do CSV de projetos.
    Usa cache para melhorar performance.
    """
    global _categorical_values_cache
    if _categorical_values_cache is not None:
        return _categorical_values_cache
    
    data_path = 'ml_model/data/projetos.csv'
    df = pd.read_csv(data_path)
    values = {}
    for field, col in CATEGORICAL_FIELDS:
        values[field] = sorted(df[col].dropna().unique().tolist())
    _categorical_values_cache = values
    return values

def fuzzy_match(value, valid_options, cutoff=0.6):
    """
    Faz um fuzzy match para encontrar o valor mais próximo em uma lista de opções válidas.
    """
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    matches = difflib.get_close_matches(
        value, [str(opt).strip().lower() for opt in valid_options], n=1, cutoff=cutoff
    )
    if matches:
        for opt in valid_options:
            if opt.strip().lower() == matches[0]:
                return opt
    return None

def normalize_project_data(project_data):
    """
    Normaliza todos os campos do projeto:
    - Converte textos com palavras extras (ex: 'meses', 'entregas') em números puros.
    - Valida e normaliza valores categóricos via fuzzy.
    """
    import unicodedata

    def strip_accents(text):
        """Remove acentuação de um texto."""
        if not isinstance(text, str):
            return text
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    def find_original(raw_value, valid_options):
        """Procura o valor original no CSV que corresponde ao match (sem acento)."""
        for option in valid_options:
            if strip_accents(option).lower() == strip_accents(raw_value).lower():
                return option
        return None

    # Normaliza duração
    dur = project_data.get('duracao_meses', '')
    if isinstance(dur, str):
        dur = dur.lower().replace('meses', '').replace('mês', '').strip()
    try:
        project_data['duracao_meses'] = int(float(dur))
    except Exception:
        project_data.pop('duracao_meses', None)

    # Normaliza orçamento, detecta mil, milhão, k
    orc = str(project_data.get('orcamento', '')).lower().replace('r$', '').replace('reais', '').strip()
    mult = 1
    if 'milhao' in strip_accents(orc) or 'milhões' in strip_accents(orc):
        mult = 1_000_000
        orc = orc.replace('milhao', '').replace('milhoes', '').strip()
    elif 'mil' in orc:
        mult = 1_000
        orc = orc.replace('mil', '').strip()
    elif 'k' in orc:
        mult = 1_000
        orc = orc.replace('k', '').strip()
    match = re.search(r'\d+[\.,]?\d*', orc)
    if match:
        val = float(match.group(0).replace(',', '.')) * mult
        project_data['orcamento'] = val
    else:
        project_data.pop('orcamento', None)

    # Normaliza entregas
    ent = project_data.get('entregas', '')
    if isinstance(ent, str):
        ent = ent.lower().replace('entregas', '').replace('entrega', '').strip()
    try:
        project_data['entregas'] = int(float(ent))
    except Exception:
        project_data.pop('entregas', None)

    # Normaliza tamanho da equipe
    eq = project_data.get('tamanho_equipe', '')
    if isinstance(eq, str):
        eq = eq.lower().replace('pessoas', '').replace('pessoa', '').replace('equipe', '').strip()
    try:
        project_data['tamanho_equipe'] = int(float(eq))
    except Exception:
        project_data.pop('tamanho_equipe', None)

    # Normaliza recursos disponíveis
    rec = project_data.get('recursos_disponiveis', '')
    rec_map = {
        'baixo': 0, 'baixa': 0, '0': 0, 'baixos': 0, 'baixas': 0, 'Baixo':0, 'Baixa':0, 'Baixos':0, 'Baixas': 0,
        'medio': 1, 'médio': 1, 'média': 1, '1': 1, 'médios': 1, 'medios': 1, 'médias': 1, 'medias': 1, 'Médio': 1, 'Medio':1,
        'Médios':1, 'Medios':1, 'Média':1, 'media':1, 'Media':1, 'Médias':1, 'Medias':1,
        'alto': 2, 'alta': 2, '2': 2, 'altos': 2, 'altas': 2, 'Alto':2, 'Alta':2, 'Altos':2, "Altas":2
    }
    try:
        rec_int = int(rec)
        if rec_int in [0, 1, 2]:
            project_data['recursos_disponiveis'] = rec_int
        else:
            project_data.pop('recursos_disponiveis', None)
    except (ValueError, TypeError):
        rec_norm = strip_accents(str(rec).lower().strip())
        if rec_norm in rec_map:
            project_data['recursos_disponiveis'] = rec_map[rec_norm]
        else:
            project_data.pop('recursos_disponiveis', None)

    # Validação fuzzy para campos categóricos
    cat_values = get_categorical_field_values()
    fuzzy_fields = ['tipo_projeto', 'departamento', 'complexidade', 'metodologia', 'risco']
    project_data['__invalid_fields'] = {}

    for field in fuzzy_fields:
        val = project_data.get(field, None)
        if not val or str(val).strip() == "":
            project_data[field] = None
            continue

        match = fuzzy_match(val, cat_values[field])
        if match:
            original = find_original(match, cat_values[field])
            project_data[field] = original if original else match
        else:
            project_data['__invalid_fields'][field] = val
            project_data[field] = None

    return project_data

def predict_project_success(project_data):
    """
    Envia os dados do projeto para a API de previsão.
    Calcula ano, mês e dia da semana baseado na data de início.
    """
    data_inicio = project_data.get('data_inicio')
    if data_inicio:
        try:
            dt = datetime.strptime(data_inicio, '%d/%m/%Y')
        except Exception:
            dt = datetime.now()
    else:
        dt = datetime.now()

    ano_inicio, mes_inicio, dia_semana = dt.year, dt.month, dt.isoweekday()

    payload = {
        'ano_inicio': ano_inicio,
        'mes_inicio': mes_inicio,
        'dia_semana': dia_semana,
    }

    for k, v in project_data.items():
        if k not in ['data_inicio', '__invalid_fields'] and v not in (None, '', []):
            payload[k] = v

    print("[DEBUG] Payload final:", json.dumps(payload, indent=2, ensure_ascii=False))

    response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()

def get_missing_fields(project_data):
    """
    Verifica quais campos obrigatórios ainda estão ausentes ou inválidos.
    """
    required_fields = [
        'duracao_meses', 'orcamento', 'entregas', 'tamanho_equipe',
        'recursos_disponiveis', 'data_inicio',
        'tipo_projeto', 'departamento', 'complexidade',
        'metodologia', 'risco'
    ]

    missing = []

    for f in required_fields:
        val = project_data.get(f, None)
        if val in (None, '', []):
            missing.append(f)

    if '__invalid_fields' in project_data and project_data['__invalid_fields']:
        missing.extend(list(project_data['__invalid_fields'].keys()))

    return missing


def format_prediction_response(prediction, project_data):
    """
    Formata o resultado final da previsão, mantendo os emojis e o estilo original.
    Usa o sucesso que já vem da API.
    """
    prob = prediction['probabilidade_sucesso']
    sucesso = prediction['sucesso']  

    dados = (
        f"Duração: {project_data['duracao_meses']} meses\n"
        f"Orçamento: R$ {project_data['orcamento']:,.2f}\n"
        f"Entregas: {project_data.get('entregas', '-')}\n"
        f"Equipe: {project_data['tamanho_equipe']} pessoas\n"
        f"Recursos: {['Baixo', 'Médio', 'Alto'][int(project_data['recursos_disponiveis'])]}\n"
        f"Data de início: {project_data.get('data_inicio', '-')}\n"
        f"Tipo: {project_data.get('tipo_projeto', '-')}\n"
        f"Departamento: {project_data.get('departamento', '-')}\n"
        f"Complexidade: {project_data.get('complexidade', '-')}\n"
        f"Metodologia: {project_data.get('metodologia', '-')}\n"
        f"Risco: {project_data.get('risco', '-')}\n"
    )

    return f"""
🎯 Previsão de Sucesso do Projeto

{dados}

📊 Resultado: {'✅ SUCESSO' if sucesso else '❌ FRACASSO'} 📈 Probabilidade: {prob:.1%}
"""

