# Detecção de fraudes em transações de cartão de crédito

Projeto de classificação binária sobre um dataset **extremamente desbalanceado**: menos de 0,2% das transações são fraudes. O foco não é "acertar muito", e sim **encontrar as fraudes sem sobrecarregar a operação com falsos alarmes**.

Notebook: `deteccao_anomalias_transacoes.ipynb`

---

## 1. O problema e por que o desbalanceamento muda a avaliação

**Dados:** dataset público de transações de cartão de crédito (`creditcard.csv`), com 284.807 transações e 31 colunas:

- `Time`: segundos desde a primeira transação;
- `V1`–`V28`: componentes de uma PCA (variáveis anonimizadas por confidencialidade);
- `Amount`: valor da transação;
- `Class`: alvo (0 = legítima, 1 = fraude).

**Desbalanceamento:** `df["Class"].value_counts(normalize=True)` mostra **99,83% legítimas** e **0,17% fraudes** (aproximadamente 1 fraude a cada 580 transações).

**Consequência para a avaliação:** um "modelo" que diz *"nunca é fraude"* já teria **~99,8% de acurácia** sem detectar nada. Por isso:

- a **acurácia é descartada** como métrica principal (no relatório ela aparece como 1.00 em todos os modelos e não diferencia nenhum);
- olho para **precisão, recall e F1 da classe de fraude (classe 1)**:
  - **Recall**: das fraudes reais, quantas o modelo pegou? (fraude não detectada = prejuízo)
  - **Precisão**: dos alertas emitidos, quantos eram fraude de fato? (alerta falso = custo operacional e cliente incomodado)
  - **F1**: média harmônica das duas;
- uso **curva Precision-Recall** além da curva ROC, porque a ROC tende a parecer otimista quando a classe positiva é rara.

---

## 2. Preparação dos dados

1. **Leitura** do CSV direto da URL pública e inspeção com `head()`.
2. **Checagem do desbalanceamento** (seção acima).
3. **Transformações no valor da transação:**
   - `Amount_log = log1p(Amount)`: reduz a assimetria de uma variável com cauda muito longa;
   - `Amount_scaled = StandardScaler(Amount)`: coloca o valor numa escala comparável às demais variáveis.
4. **Separação treino/teste:** 70% / 30%, com `stratify=y` e `random_state=42`. A estratificação mantém a proporção de fraudes nos dois conjuntos (o teste tem 85.443 linhas, sendo **148 fraudes**).
5. **Técnicas de reamostragem exploradas:**
   - *Undersampling*: todas as fraudes + mesma quantidade de transações normais sorteadas (`df_under`);
   - *Oversampling com SMOTE*: fraudes sintéticas geradas por interpolação (`X_res`, `y_res`).
   - ⚠️ Esses dois conjuntos foram **criados, mas não usados para treinar nenhum modelo** no notebook. O balanceamento foi tratado nos modelos via `class_weight="balanced"` (Random Forest) e `scale_pos_weight=10` (XGBoost).

---

## 3. Comparação entre os modelos (classe de fraude, conjunto de teste)

| Modelo | Precisão | Recall | F1 |
|---|---|---|---|
| Regressão Logística (baseline, `max_iter=1000`) | 0,84 | 0,66 | 0,73 |
| Regressão Logística com limiar 0,3 | 0,79 | 0,71 | 0,75 |
| Random Forest (50 árvores, `max_depth=10`, `class_weight="balanced"`) | 0,84 | 0,76 | 0,79 |
| **XGBoost** (`scale_pos_weight=10`) | **0,94** | **0,78** | **0,85** |

Além disso, a regressão logística teve **AUC-ROC = 0,927**.

**Leitura dos resultados:**

- O **XGBoost foi o melhor em precisão, recall e F1** ao mesmo tempo.
- Passar da regressão logística para modelos de árvores aumentou o recall (0,66 → 0,76/0,78) sem perder precisão.
- Com só 148 fraudes no teste, uma diferença de 0,02 em recall equivale a **poucas transações**; a diferença entre Random Forest e XGBoost em recall é pequena, mas a de precisão (0,84 → 0,94) é mais clara.
- Foi rodado um `GridSearchCV` (`scoring="recall"`, `cv=3`) para o XGBoost, com melhor resultado `max_depth=5, n_estimators=100`. O modelo ajustado **não foi reavaliado no teste** no notebook.

---

## 4. Limiar de decisão e SHAP

**Limiar de decisão.** Por padrão, o modelo classifica como fraude quando a probabilidade passa de 0,5. Testei o limiar **0,3** sobre as probabilidades da regressão logística (`predict_proba`):

- recall: 0,66 → **0,71**
- precisão: 0,84 → **0,79**
- F1: 0,73 → **0,75**

Baixar o limiar faz o modelo "desconfiar mais": pega mais fraudes, ao custo de mais alertas falsos. O valor 0,3 foi escolhido manualmente; a escolha ideal depende do **custo de uma fraude não detectada vs. o custo de um alerta falso** no negócio.

**SHAP.** Usei `shap.Explainer` sobre o XGBoost com as 100 primeiras linhas do teste e `shap.plots.bar` para ver a **contribuição média absoluta de cada variável** para as previsões. O gráfico complementa o `feature_importances_` do XGBoost (que indica o quanto a variável é usada pelo modelo) mostrando o **impacto médio nas previsões**.

> 📌 Preencher aqui as variáveis que apareceram no topo do gráfico SHAP do seu notebook (ex.: "V14, V4, ..."), e o que isso sugere. As variáveis `V1`–`V28` são anônimas (PCA), então a interpretação é sobre **quais componentes pesam**, não sobre o significado de negócio de cada uma.

---

## 5. Como executar

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib
jupyter notebook deteccao_anomalias_transacoes.ipynb
```

O notebook baixa o dataset automaticamente da URL pública, então não precisa de arquivo local.
