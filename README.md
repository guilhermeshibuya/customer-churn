# Customer Churn Prediction

Projeto de MLOps para prever a probabilidade de churn de clientes de telecomunicações. O fluxo cobre carregamento e preparação dos dados, treinamento de um pipeline scikit-learn, avaliação em conjunto de teste e rastreamento/registro dos experimentos com MLflow.

O dataset utilizado é o público **Telco Customer Churn**. O arquivo deve estar em:

```text
data/raw/customer-churn-dataset.csv
```

## Visão geral

O pipeline executa as seguintes etapas:

1. Carrega o CSV e trata `TotalCharges` como numérico.
2. Separa os dados em treino, validação e teste.
3. Remove o identificador `customerID` e codifica o alvo `Churn` (`No` = 0, `Yes` = 1).
4. Aplica `OneHotEncoder` nas variáveis categóricas e `StandardScaler` nas numéricas.
5. Treina o classificador definido no arquivo de configuração.
6. Avalia precision, recall e F1 no conjunto de teste.
7. Registra parâmetros, métricas e o modelo no MLflow.
8. Salva o pipeline treinado em `models/customer_churn_pipeline.joblib`.

O pré-processamento e o classificador ficam no mesmo `Pipeline`, garantindo consistência entre treinamento e inferência.

## Estrutura do projeto

```text
customer-churn/
├── configs/
│   └── train.yaml                 # Configuração do experimento
├── data/
│   ├── raw/                       # Dataset original (ignorado pelo Git)
│   └── processed/                 # Dados processados (ignorado pelo Git)
├── docs/
│   └── PLANNING.md                # Especificação e planejamento do projeto
├── models/                        # Modelos persistidos localmente
├── notebooks/
│   └── exploration.ipynb          # Análise exploratória
├── src/customer_churn/
│   ├── __init__.py                # Ponto de entrada (CLI)
│   ├── config.py                  # Carregamento e validação da configuração
│   ├── data.py                    # Leitura, split e encoding do alvo
│   ├── validation.py              # Validação de schema
│   ├── preprocessing.py           # Pré-processamento e pipeline
│   ├── train.py                   # Criação, treino e persistência do modelo
│   ├── evaluate.py                # Métricas de avaliação
│   ├── tracking.py                # Integração com MLflow
│   ├── pipeline.py                # Orquestração do treinamento
│   └── models/factory.py          # Registry de modelos suportados
├── tests/                         # Espaço reservado para testes automatizados
└── pyproject.toml                 # Dependências e entry point
```

## Pré-requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Dataset em `data/raw/customer-churn-dataset.csv`

## Instalação

Na raiz do projeto:

```bash
uv sync
```

## Configuração

O experimento é configurado em `configs/train.yaml`:

```yaml
raw_data_path: data/raw/customer-churn-dataset.csv
processed_data_path: data/processed
model_output_path: models/customer_churn_pipeline.joblib

target_column: Churn
id_column: customerID

test_size: 0.1
validation_size: 0.2
random_state: 42
cv_folds: 5

tracking_uri: http://127.0.0.1:5000
experiment_name: customer-churn

model:
  name: random_forest
  params:
    n_estimators: 200
    random_state: 42
    class_weight: balanced
```

Modelos disponíveis em `src/customer_churn/models/factory.py`:

- `random_forest`
- `logistic_regression`

## Como rodar um experimento

### 1. Inicie o servidor MLflow

Em um terminal, na raiz do projeto:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

O `tracking_uri` do YAML aponta para esse endereço. O backend SQLite é necessário para usar o Model Registry.

### 2. Execute o treinamento

Em outro terminal:

```bash
uv run customer-churn --config configs/train.yaml
```

Saída esperada ao final:

```text
Successfully registered model 'customer-churn'.
Created version '1' of model 'customer-churn'.
Evaluation Results: {'precision': ..., 'recall': ..., 'f1_score': ...}
```

### 3. Abra a interface do MLflow

Com o servidor rodando, acesse:

```text
http://127.0.0.1:5000
```

Na UI é possível comparar runs, consultar parâmetros/métricas, baixar artefatos e gerenciar as versões do modelo registrado `customer-churn`.

Se quiser apenas visualizar os experimentos já gravados:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Rodando outro experimento

Crie outro arquivo YAML alterando modelo, hiperparâmetros ou `experiment_name` e execute:

```bash
uv run customer-churn --config configs/train_logistic.yaml
```

Exemplo de `configs/train_logistic.yaml`:

```yaml
raw_data_path: data/raw/customer-churn-dataset.csv
processed_data_path: data/processed
model_output_path: models/customer_churn_logistic.joblib

target_column: Churn
id_column: customerID

test_size: 0.1
validation_size: 0.2
random_state: 42
cv_folds: 5

tracking_uri: http://127.0.0.1:5000
experiment_name: customer-churn

model:
  name: logistic_regression
  params:
    max_iter: 1000
    class_weight: balanced
```

## Usando o modelo registrado

Carregar uma versão específica:

```python
import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/customer-churn/1")
predictions = model.predict(X)
```

Carregar a partir de uma run:

```python
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```

Servir o modelo como API:

```bash
uv run mlflow models serve \
  -m "models:/customer-churn/1" \
  --port 5001 \
  --env-manager local
```

## Resultado de referência

Execução baseline com `random_forest` e a configuração padrão:

| Métrica   | Valor  |
| --------- | ------ |
| Precision | 0.594  |
| Recall    | 0.457  |
| F1-score  | 0.517  |

Os valores podem variar conforme a versão do dataset, o split e os hiperparâmetros.

## Artefatos locais

- `mlflow.db`: banco com runs, parâmetros, métricas e registro de modelos.
- `mlruns/`: artefatos das runs (modelos, assinaturas, arquivos de configuração).
- `models/customer_churn_pipeline.joblib`: cópia local do pipeline treinado.

`mlflow.db`, `mlruns/` e o conteúdo de `data/` estão no `.gitignore`.

## Próximos passos

- Implementar testes automatizados em `tests/`.
- Adicionar validação de dados ao fluxo principal.
- Definir critérios de aprovação e promoção de modelos no Model Registry.
- Criar API de inferência com FastAPI.
- Configurar CI e empacotamento com Docker.

A especificação completa está em [`docs/PLANNING.md`](docs/PLANNING.md).
