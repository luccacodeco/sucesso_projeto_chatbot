import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score)
import joblib


def train_model():
    """
    Treina um modelo Random Forest para prever o sucesso de projetos.
    Realiza engenharia de variáveis de data, pré-processamento estruturado,
    ajuste de hiperparâmetros via RandomizedSearchCV e busca de threshold ótimo.
    Salva o pipeline final com threshold.
    """

    # Caminho fixo do arquivo de entrada
    data_path = 'ml_model/data/projetos.csv'
    df = pd.read_csv(data_path, parse_dates=['data_inicio'])
    print(f"Projetos carregados: {len(df)} linhas")

    # Criação de colunas derivadas de data
    df['ano_inicio'] = df['data_inicio'].dt.year
    df['mes_inicio'] = df['data_inicio'].dt.month
    df['dia_semana'] = df['data_inicio'].dt.dayofweek

    # Colunas numéricas e categóricas
    numeric_features = [
        'duracao_meses', 'orcamento', 'entregas',
        'tamanho_equipe', 'recursos_disponiveis',
        'ano_inicio', 'mes_inicio', 'dia_semana'
    ]

    categorical_features = [
        'tipo_projeto', 'departamento',
        'complexidade', 'metodologia', 'risco'
    ]

    target = 'sucesso'
    X = df[numeric_features + categorical_features]
    y = df[target]

    # Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pré-processamento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(
                drop='first',
                handle_unknown='ignore',
                sparse_output=False
            ), categorical_features)
        ]
    )

    # Pipeline Random Forest
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            oob_score=True
        ))
    ])

    # Parâmetros para busca aleatória
    param_dist = {
        'clf__n_estimators': [200, 500, 800, 1000],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 5],
        'clf__max_features': ['sqrt', 'log2'],
        'clf__class_weight': [None, 'balanced_subsample']
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=30,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Treinamento
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    print(f"\nMelhores parâmetros: {search.best_params_}")

    # Ajuste do threshold
    probs = best_rf.predict_proba(X_test)[:, 1]
    best_thresh, best_acc = 0.5, 0
    for t in np.linspace(0.3, 0.7, 41):
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_test, preds)
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    print(f"Melhor threshold: {best_thresh:.2f} → Acurácia: {best_acc:.2%}")

    # Métricas finais
    y_pred = (probs >= best_thresh).astype(int)
    acc_final = accuracy_score(y_test, y_pred)
    auc_final = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nMétricas no teste:")
    print(f"Acurácia: {acc_final:.2%}")
    print(f"ROC AUC: {auc_final:.2%}")
    print(f"Precisão: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, digits=4))

    # Matriz de Confusão 
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=['Atual 0', 'Atual 1'],
        columns=['Previsto 0', 'Previsto 1']
    )
    print("\nMatriz de Confusão:")
    print(cm_df)

    # Caminho para salvar
    output_path = 'ml_model/model.pkl'
    joblib.dump({'model': best_rf, 'threshold': best_thresh}, output_path)
    print(f"\nModelo salvo em: {output_path}")

    return best_rf, best_thresh


if __name__ == '__main__':
    train_model()
