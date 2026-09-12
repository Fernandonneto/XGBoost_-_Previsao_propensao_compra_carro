# 🧠 XGBoost: Previsão de Propensão à Compra de Carros

Projeto desenvolvido no Módulo XGBoost do curso de Cientista de Dados da EBAC, com foco na aplicação do algoritmo XGBoost para prever a propensão de clientes à compra de carros.

A partir de uma base contendo informações demográficas e financeiras dos clientes, foram realizadas etapas de preparação dos dados, análise de correlação, treinamento do modelo, avaliação de desempenho e análise da importância das variáveis (Feature Importance).

## 🎯 Objetivo do Projeto

O objetivo é construir um modelo de Machine Learning supervisionado capaz de prever se um cliente possui propensão a realizar a compra de um carro.

A análise busca responder:

    Quais características dos clientes estão mais associadas à decisão de compra e quais variáveis apresentam maior importância para o modelo?

Os resultados podem apoiar estratégias comerciais como:

- identificação de potenciais compradores;
- segmentação de clientes;
- direcionamento de campanhas;
- priorização de abordagens comerciais;
- tomada de decisão orientada por dados.

 ## 🗂️ Base de Dados

A base utilizada é a **CARRO_CLIENTES.csv**, contendo informações relacionadas ao perfil dos clientes e à ocorrência de compra.

Entre as principais variáveis estão:

- `User ID` — identificador do cliente;
- `Gender` — gênero;
- `Age` — idade;
- `AnnualSalary` — salário anual;
- `Purchased` — variável alvo indicando a realização da compra.

A variável `Purchased` foi definida como variável dependente (**y**).

## 🧩 ETAPA 1 — Preparação dos Dados

### 📥 Carregamento da base

A base foi importada utilizando Pandas, com posterior inspeção dos registros e da estrutura dos dados.

    base = pd.read_csv('CARRO_CLIENTES.csv', delimiter=',')

Também foram verificadas as informações relacionadas aos tipos de dados.

### 🔎 Tipos de dados

Durante a análise, foi identificada a variável categórica:

    Gender

As demais variáveis relevantes para a modelagem estavam representadas numericamente.

### 🧹 Verificação de dados faltantes

Foi realizada uma análise dos valores nulos presentes na base.

O resultado indicou ausência de dados faltantes, não sendo necessária a aplicação de técnicas de imputação ou exclusão de registros.

### 🆔 Remoção do identificador

A coluna:

    User ID

foi removida antes da modelagem.

Apesar de ser numérica, essa variável representa apenas um identificador único dos clientes e não possui significado comportamental ou estatístico relevante para a previsão.

Sua permanência poderia introduzir ruído desnecessário no modelo.

## 🔤 ETAPA 2 — Codificação da Variável Categórica

Para permitir a utilização da variável Gender no XGBoost, foi aplicado o LabelEncoder.

    label_encoder = LabelEncoder()

    base['Gender'] = label_encoder.fit_transform(base['Gender'])

Com isso, a variável categórica passou a ser representada numericamente, tornando a base compatível com o algoritmo.

## 📊 ETAPA 3 — Análise de Correlação

Foi construída uma matriz de correlação para analisar a relação linear entre as variáveis.

As maiores correlações com a variável alvo `Purchased` foram:

| Variável | Correlação com `Purchased` |
---      | ---
`Age`      | 0.62
`AnnualSalary`| 0.36
`Gender`   | -0.05

### 🔎 Principais observações

A variável `Age` apresentou a maior associação linear com a decisão de compra, seguida por `AnnualSalary`.

Já `Gender` apresentou uma correlação próxima de zero, indicando uma relação linear praticamente inexistente com a variável alvo.

    Importante: correlação não representa causalidade e mede apenas a associação linear entre as variáveis.

## ✂️ ETAPA 4 — Separação dos Dados

A variável:

    Purchased

foi definida como variável dependente (**y**), enquanto as demais variáveis selecionadas compuseram as variáveis independentes (**X**).

A base foi dividida em:

- 80% para treinamento;
- 20% para teste;
- `random_state=42` para garantir reprodutibilidade.

---
    
    X = base.drop('Purchased', axis=1)
    y = base['Purchased']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

### ⚙️ Escopo da modelagem

O objetivo do exercício foi aplicar o XGBoost e interpretar suas variáveis mais relevantes.

Por esse motivo, não foram realizadas:

- busca exaustiva de hiperparâmetros;
- Grid Search;
- Cross Validation;
- comparação com outros algoritmos;
- técnicas adicionais de balanceamento.

## 🤖 ETAPA 5 — Modelagem com XGBoost

Foi utilizado o:

**XGBClassifier**

com os seguintes hiperparâmetros:

    xgb_model = xgb.XGBClassifier(
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

### 🔧 Hiperparâmetros utilizados

`max_depth=6` \
Define a profundidade máxima das árvores. Valores maiores permitem capturar padrões mais complexos, mas podem aumentar o risco de overfitting.

`n_estimators=100` \
Define a quantidade de árvores utilizadas no processo de boosting.

`learning_rate=0.1` \
Controla a contribuição de cada nova árvore durante o aprendizado. Valores menores tornam o aprendizado mais gradual.

## 🔮 ETAPA 6 — Previsões e Probabilidades

Após o treinamento, foram realizadas previsões sobre o conjunto de teste.

Foram obtidos:

**Classificação**

    y_pred = xgb_model.predict(X_test)

**Probabilidades**

    y_pred_prob = xgb_model.predict_proba(X_test)

A utilização de `predict_proba()` permite observar a probabilidade estimada pelo modelo para cada classe, além da classificação final.

## 📈 ETAPA 7 — Avaliação do Modelo

O desempenho foi avaliado utilizando:

- `Accuracy`
- `Precision`
- `Recall`
- `F1-Score`
- `Matriz de Confusão`

### 🏆 Resultados

O modelo apresentou **91% de acurácia geral** no conjunto de teste.

Para a classe de compradores, foram observados:

- **Precision:** 93%;
- **Recall:** 86%;
- **F1-Score:** 0,89.

A matriz de confusão apresentou apenas 6 falsos positivos, indicando uma boa capacidade do modelo de evitar classificar incorretamente clientes que não realizaram a compra.

### 💼 Interpretação de Negócio

A Precision de 93% significa que, entre os clientes classificados pelo modelo como potenciais compradores, aproximadamente 93% realmente pertenciam à classe de compradores.

Já o Recall de 86% indica que o modelo conseguiu identificar aproximadamente 86% dos compradores reais presentes no conjunto de teste.

Esses resultados demonstram um desempenho elevado para o escopo proposto.

## ⭐ ETAPA 8 — Feature Importance

Após o treinamento, foi analisada a importância das variáveis utilizando o Gain do XGBoost.

Os resultados foram:

| Feature | Importance |
---      | ---
`Age` | 4.56
`AnnualSalary` | 2.47
`Gender` | 0.74

### 🔎 Principais insights

A variável `Age` apresentou a maior importância para as decisões realizadas pelo modelo, seguida por `AnnualSalary`.

`Gender` apresentou importância consideravelmente menor.

Os resultados foram consistentes com a análise de correlação, que também havia identificado `Age` e `AnnualSalary` como as variáveis com maior associação com `Purchased`.

Entretanto, os dois métodos possuem interpretações diferentes:

- **Correlação:** mede a associação linear entre uma variável e o alvo;
- **Feature Importance:** mede a contribuição da variável para as decisões realizadas pelo modelo de árvores.

Portanto, a proximidade dos resultados reforça a relevância de `Age` e `AnnualSalary`, mas não significa que os dois métodos estejam medindo exatamente a mesma coisa.

## 🛠️ Tecnologias Utilizadas

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Jupyter Notebook**

## 🎯 Competências Demonstradas

- Data Loading;
- Data Cleaning;
- Data Preparation;
- Análise de tipos de dados;
- Tratamento de variáveis categóricas;
- Label Encoding;
- Análise de Correlação;
- Train/Test Split;
- Machine Learning supervisionado;
- XGBoost Classifier;
- Definição de hiperparâmetros;
- Classificação binária;
- Predição de probabilidades;
- Accuracy;
- Precision;
- Recall;
- F1-Score;
- Matriz de Confusão;
- Feature Importance;
- Interpretação de modelos de Machine Learning.

## ✅ Conclusão

O projeto apresentou um fluxo completo de aplicação do XGBoost para classificação, desde a preparação e análise dos dados até a avaliação do modelo e identificação das principais variáveis preditivas.

O modelo alcançou 91% de acurácia, com F1-Score de 0,89 para a classe de compradores, apresentando bom desempenho dentro do escopo proposto.

A análise de Feature Importance destacou `Age` e `AnnualSalary` como as variáveis de maior relevância para as decisões do modelo, resultados que também foram coerentes com a análise inicial de correlação.

O projeto demonstra, na prática, a aplicação de Machine Learning, análise estatística e interpretação de modelos para apoiar problemas de segmentação e previsão em um contexto comercial.