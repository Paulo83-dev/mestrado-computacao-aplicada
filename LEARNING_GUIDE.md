# 📖 Guia Completo de Aprendizado - Machine Learning

Um roteiro detalhado para dominar Machine Learning através das aulas do Prof. Francisco de Assis Boldt.

---

## 🎯 Objetivos Gerais do Curso

Ao final deste curso você será capaz de:

- ✅ Entender a **matemática por trás** dos algoritmos de ML
- ✅ **Implementar do zero** algoritmos de regressão e classificação
- ✅ **Avaliar modelos** corretamente com métricas apropriadas
- ✅ **Otimizar hiperparâmetros** usando Grid Search
- ✅ **Evitar armadilhas** como overfitting e data leakage
- ✅ **Aplicar a prática** em projetos reais (Titanic)
- ✅ **Agrupar dados** com algoritmos de clustering

---

## 📚 Módulo 1: Fundamentos de Regressão
**Duração:** 1-2 semanas | **Conceitos-chave:** Regressão Linear, Métricas, MSE/MAE

### Aula 01a: Regressão Linear - Os Primeiros Passos
**Arquivo:** `machine-learning/aula01a_regression.ipynb`

**O que você aprenderá:**
- 🔍 Carregar e explorar dados (EDA - Exploratory Data Analysis)
- 📐 Implementar regressão linear **do zero**
- 📊 Visualizar relações entre variáveis
- 📈 Entender a equação da reta: `y = ax + b`
- 📉 Calcular métricas de erro: MAE, MSE, RMSE

**Dataset:** Diabetes (442 amostras, 10 features)

**Saída esperada:** Compreensão intuitiva de como uma reta "melhor encaixa" dados contínuos.

**Exercício Prático:**
```python
# Depois da aula, tente:
# 1. Carregar o dataset de diabetes
# 2. Plotar outra coluna (não apenas BMI)
# 3. Calcular MAE, MSE, RMSE manualmente
# 4. Comparar diferentes valores de 'a' e 'b'
```

---

### Aula 02a: Regressão Linear Avançada
**Arquivo:** `machine-learning/aula02a_regression.ipynb`

**O que você aprenderá:**
- 🔢 Regressão com **múltiplas features** (multidimensional)
- 📐 Álgebra matricial para resolver o problema
- ⚙️ Uso do `sklearn.linear_model.LinearRegression`
- 🎯 Comparação de modelos diferentes

**Saída esperada:** Capacidade de treinar modelos com múltiplas variáveis.

---

### Aula 02b: Equação Normal
**Arquivo:** `machine-learning/aula02b - normal equation.ipynb`

**O que você aprenderá:**
- 🔬 Derivação matemática da **pseudoinversa**
- 📊 Solução analítica vs otimização iterativa
- 🚀 Quando usar equação normal vs gradiente descendente

**Saída esperada:** Compreensão da geometria de mínimos quadrados.

---

### Aula 02c: Comparação de Regressores
**Arquivo:** `machine-learning/aula02c - comparações de regressores.ipynb`

**O que você aprenderá:**
- 🏋️ LinearRegression vs Ridge vs Lasso vs ElasticNet
- 📏 Regularização (L1 e L2) para evitar overfitting
- 📊 Comparação de desempenho entre modelos
- 🎯 Quando usar cada técnica

**Saída esperada:** Saber escolher o regressor mais apropriado para cada problema.

---

## 📚 Módulo 2: Introdução à Classificação
**Duração:** 1-2 semanas | **Conceitos-chave:** Classificação, Limiar, Métricas

### Aula 03a: Classificação Linear
**Arquivo:** `machine-learning/aula03a_classificação.ipynb`

**O que você aprenderá:**
- 🎯 Transformar regressão em classificação via **limiar (threshold)**
- 📊 Análise de correlações para seleção de features
- 🏥 Dataset real: Breast Cancer (diagnóstico de câncer)
- ✅ Acurácia como métrica inicial

**Dataset:** Breast Cancer (569 amostras, 30 features)

**Saída esperada:** Entender que classificação é "apenas" regressão + limiar.

**Exercício Prático:**
```python
# Experimente diferentes thresholds:
# Padrão: 0.5
# Sensível (mais falsos positivos): 0.3
# Específico (menos falsos positivos): 0.7
# Veja como a acurácia muda!
```

---

### Aula 03b: Normalização de Features
**Arquivo:** `machine-learning/aula03b - normalização.ipynb`

**O que você aprenderá:**
- 📏 **StandardScaler:** Normalizar para média=0, desvio=1
- 📦 **MinMaxScaler:** Escalar para [0, 1]
- 🎯 Por que normalizar melhora performance
- 🔁 Usar Pipelines para evitar data leakage

**Saída esperada:** Dados prontos e padronizados para treino.

---

## 📚 Módulo 3: Validação e Otimização
**Duração:** 1-2 semanas | **Conceitos-chave:** Validação, Otimização, Pipelines

### Aula 04a: K-Nearest Neighbors (KNN)
**Arquivo:** `machine-learning/aula04a_KNN.ipynb`

**O que você aprenderá:**
- 🔍 Algoritmo KNN (votos dos vizinhos mais próximos)
- 📏 Métricas de distância (Euclidiana, Manhattan)
- 🎯 Seleção do hiperparâmetro `k`
- ⚡ Trade-off entre bias e variância

**Dataset:** Iris (150 amostras, 4 features)

**Saída esperada:** Primeiro algoritmo "lazy learning" aprendido.

**Exercício Prático:**
```python
# Teste diferentes valores de k:
# k=1: muito sensível a ruído
# k=5: equilíbrio
# k=15: muito generalista
# Plote acurácia vs k
```

---

### Aula 04b: Pipelines
**Arquivo:** `machine-learning/aula04b - Pipelines.ipynb`

**O que você aprenderá:**
- 🔗 Encadear preprocessamento + modelo
- ⚠️ Evitar **data leakage** (vazamento de dados)
- 🔄 Reproduzibilidade: treino → validação → teste
- 🎯 Boas práticas em ML

**Saída esperada:** Workflow seguro e reproduzível.

---

### Aula 04c: Validação Cruzada (Cross-Validation)
**Arquivo:** `machine-learning/aula04c - cross validation.ipynb`

**O que você aprenderá:**
- 🔄 **K-Fold CV:** Dividir dados em k partes
- 📊 Estimativa confiável do desempenho real
- 🎯 Evitar overfitting e underfitting
- 📈 Plotar curvas de aprendizado

**Saída esperada:** Confiança na avaliação do modelo.

**Exercício Prático:**
```python
# Compare:
# - Train/Test simples (80/20)
# - 5-Fold Cross-Validation
# - 10-Fold Cross-Validation
# Qual é mais estável?
```

---

### Aula 04d: Grid Search
**Arquivo:** `machine-learning/aula04d - GridSearch.ipynb`

**O que você aprenderá:**
- 🔍 **GridSearchCV:** Buscar melhores hiperparâmetros
- 📊 Automatizar otimização
- 🎯 Integração com Pipelines
- ⏱️ Paralelização para velocidade

**Saída esperada:** Otimizar modelos automaticamente.

---

## 📚 Módulo 4: Árvores e Ensembles
**Duração:** 2 semanas | **Conceitos-chave:** Árvores, Ensemble, Random Forest

### Aula 05: Árvores de Decisão
**Arquivos:** 
- `machine-learning/aula05_árvore_de_decisão_atributos_discretos.ipynb`
- `machine-learning/aula05b - árvore de decisão - atributos contínuos.ipynb`

**O que você aprenderá:**
- 🌳 Construir árvores de decisão do zero
- 📊 Impureza: Gini vs Entropia
- ✂️ Poda para evitar overfitting
- 🔢 Lidar com features discretas vs contínuas

**Saída esperada:** Modelos interpretáveis e poderosos.

---

### Aula 06a: Ensembles
**Arquivo:** `machine-learning/aula06a_ensembles.ipynb`

**O que você aprenderá:**
- 🌲 **Random Forest:** Múltiplas árvores votam
- 🚀 **Gradient Boosting:** Árvores sequenciais que aprendem erros
- 🗳️ **Votação e Stacking:** Combinar modelos diferentes
- 📊 Feature Importance

**Saída esperada:** Modelos de alta performance em competições Kaggle.

---

## 📚 Módulo 5: Projeto Prático
**Duração:** 1-2 semanas | **Conceito-chave:** Aplicação Real

### Aula 07a: Projeto Titanic
**Arquivo:** `machine-learning/aula07a_Titanic.ipynb`

**O que você aprenderá:**
- 🧹 **Limpeza de dados** (missing values, outliers)
- 🔧 **Feature Engineering** (criar novas features)
- 📊 Exploração aprofundada com EDA
- 🎯 Aplicar TODO o conhecimento anterior
- 📈 Submeter predictions no Kaggle

**Dataset:** Titanic (891 amostras, 11 features)

**Saída esperada:** Seu primeiro projeto de ML completo!

**Milestone:** Competir no Kaggle!

---

## 📚 Módulo 6: Aprendizado Não Supervisionado
**Duração:** 1-2 semanas | **Conceitos-chave:** Clustering, Densidade

### Aula 08a: K-Means
**Arquivo:** `machine-learning/aula08a_kmeans.ipynb`

**O que você aprenderá:**
- 🎯 K-Means do zero: inicialização, atribuição, atualização
- 📊 Centróides e convergência
- 🔄 Visualização de iterações
- 🎨 Diagrama de Voronoi (fronteiras de decisão)

**Dataset:** Iris (agora SEM rótulos)

**Saída esperada:** Agrupar dados sem supervisão.

**Exercício Prático:**
```python
# Implemente K-Means você mesmo!
# 1. Inicialize centróides aleatoriamente
# 2. Calcule distâncias euclidianas
# 3. Atribua pontos ao centróide mais próximo
# 4. Atualize centróides como média dos clusters
# 5. Repita até convergência
```

---

### Aula 08b: Método do Cotovelo (Elbow Method)
**Arquivo:** `machine-learning/aula08b_kmeans_elbow.ipynb`

**O que você aprenderá:**
- 📈 Encontrar k ótimo graficamente
- 🧮 Inércia (soma de distâncias dentro de clusters)
- 📊 Plotar curva de cotovelo
- 🎯 Escolher k de forma inteligente

**Saída esperada:** Determinar número de clusters automaticamente.

---

### Aula 08c: K-Means para Seleção de Features
**Arquivo:** `machine-learning/aula08c_kmeans_as_fs.ipynb`

**O que você aprenderá:**
- 🔍 Usar clustering para reduzir dimensionalidade
- 📦 Feature Selection vs Feature Extraction
- 🚀 Acelerar treinamento removendo features redundantes

**Saída esperada:** Dados mais compactos sem perder informação.

---

### Aula 09a: DBSCAN
**Arquivo:** `machine-learning/aula09a_dbscan.ipynb`

**O que você aprenderá:**
- 🔍 Clustering baseado em **densidade**
- 🎯 Parâmetros eps (raio) e min_samples
- ✨ Encontrar clusters de **formas arbitrárias**
- 📍 Detectar outliers (pontos de ruído)

**Saída esperada:** Alternativa ao K-Means para dados complexos.

**Exercício Prático:**
```python
# Gere dados em forma de lua (sklearn.datasets.make_moons)
# Veja como:
# - KMeans falha
# - DBSCAN acerta!
```

---

### Aula 09b: Semi-Supervised Learning
**Arquivo:** `machine-learning/aula09b_semi_supervised_learning.ipynb`

**O que você aprenderá:**
- 🎯 Usar dados rotulados E não rotulados
- 🔄 Self-training e co-training
- 📊 Quando usar quando supervisado é caro
- 🚀 Boost performance com poucos labels

**Saída esperada:** Maximizar valor dos dados limitados.

---

## 🎓 Estrutura de Cada Aula

Cada notebook segue este padrão:

1. **Introdução Clara**
   - Qual é o objetivo?
   - Por que aprender isto?
   - Como se conecta com aulas anteriores?

2. **Carregamento de Dados**
   - EDA: exploração básica
   - Visualização inicial

3. **Implementação Do Zero**
   - Código em NumPy puro
   - Explicações passo a passo
   - Por quê cada linha?

4. **Uso da Biblioteca**
   - sklearn implementação
   - Comparar com implementação manual

5. **Visualização**
   - Gráficos para intuição
   - Mostrar como o modelo vê os dados

6. **Métricas e Avaliação**
   - Como medir sucesso?
   - Comparar com alternativas

7. **Discussão Pedagógica**
   - Armadilhas comuns
   - Extensões avançadas
   - Referências para aprofundar

---

## 📅 Cronograma Sugerido

### Mês 1 (Semanas 1-4)
```
Semana 1: Aulas 01a, 02a
Semana 2: Aulas 02b, 02c
Semana 3: Aulas 03a, 03b
Semana 4: Aulas 04a, 04b
```

### Mês 2 (Semanas 5-8)
```
Semana 5: Aulas 04c, 04d
Semana 6: Aulas 05, 05b
Semana 7: Aula 06a
Semana 8: Aula 07a (Projeto)
```

### Mês 3 (Semanas 9-12)
```
Semana 9: Aulas 08a, 08b
Semana 10: Aulas 08c, 09a
Semana 11: Aula 09b
Semana 12: Revisão + Projeto Extra
```

---

## 🎯 Metas de Aprendizado por Módulo

### Módulo 1 ✅
- [ ] Implementar regressão linear do zero
- [ ] Calcular MSE, MAE, RMSE
- [ ] Diferenciar LinearRegression, Ridge, Lasso
- [ ] Resolver problema usando equação normal

### Módulo 2 ✅
- [ ] Implementar classificador com limiar
- [ ] Normalizar features usando StandardScaler
- [ ] Entender importância da normalização

### Módulo 3 ✅
- [ ] Implementar KNN do zero
- [ ] Usar Pipelines para evitar data leakage
- [ ] Realizar k-fold cross-validation
- [ ] Otimizar hiperparâmetros com GridSearch

### Módulo 4 ✅
- [ ] Construir árvore de decisão manualmente
- [ ] Entender impureza (Gini/Entropia)
- [ ] Usar Random Forest e Gradient Boosting
- [ ] Interpretar feature importance

### Módulo 5 ✅
- [ ] Limpar dataset real (missing values, duplicatas)
- [ ] Fazer feature engineering criativo
- [ ] Explorar dados com EDA aprofundado
- [ ] Submeter predictions no Kaggle

### Módulo 6 ✅
- [ ] Implementar K-Means do zero
- [ ] Encontrar k ótimo com método do cotovelo
- [ ] Usar DBSCAN para clustering baseado em densidade
- [ ] Aplicar semi-supervised learning

---

## 💪 Desafios Adicionais

Depois de completar cada módulo, tente:

### Após Módulo 1
- Aplique regressão em um dataset do Kaggle
- Implemente gradient descent manualmente

### Após Módulo 2
- Classifique dados de um novo dataset
- Varie o threshold e plotar ROC curve

### Após Módulo 3
- Otimize um modelo para Kaggle
- Implemente validação cruzada do zero

### Após Módulo 4
- Compare todos os algoritmos em um dataset
- Faça stacking de modelos

### Após Módulo 5
- Complete o projeto Titanic
- Suba suas predictions no Kaggle

### Após Módulo 6
- Clusterize dados de seu interesse
- Combine clustering + classificação

---

## 📚 Leitura Complementar

### Livros Recomendados
- 📖 "Hands-On Machine Learning" - Aurélien Géron
- 📖 "An Introduction to Statistical Learning" - James, Witten, Hastie, Tibshirani
- 📖 "Deep Learning" - Goodfellow, Bengio, Courville

### Cursos Online
- 📺 [3Blue1Brown - Linear Algebra Essentials](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- 📺 [Andrew Ng - Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)

### Datasets para Praticar
- 🏠 [Housing Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- 🐶 [Dog Breed Classification](https://www.kaggle.com/competitions/dog-breed-identification)
- 🎬 [Movie Recommendations](https://www.kaggle.com/datasets/grouplens/movielens-latest-small)

---

## ✨ Próximos Passos Após o Curso

1. **Deep Learning:** TensorFlow/PyTorch para redes neurais
2. **NLP:** Processamento de linguagem natural com transformers
3. **Visão Computacional:** Detectar objetos em imagens
4. **Séries Temporais:** Prever preços de ações, weather forecasting
5. **Pesquisa:** Ler papers no arXiv e implementar algoritmos novos

---

## 🤝 Como Usar Este Guia

1. **Siga a ordem sugerida** - Cada aula constrói sobre a anterior
2. **Execute todos os exemplos** - Não só leia, FAÇA
3. **Modifique o código** - Tente coisas diferentes
4. **Tome notas** - Escreva o que aprendeu
5. **Faça os exercícios** - Praticar é essencial
6. **Tire dúvidas** - Consulte o professor, colegas, comunidade

---

> **Lembre-se:** Machine Learning é 10% teoria e 90% prática. Use suas mãos!

🚀 **Pronto para começar? Abra a Aula 01a!**
