# 🧠 Machine Learning (Mestrado em Computação Aplicada)

Repositório de aulas práticas da disciplina **Machine Learning** ministrada pelo **Prof. Francisco de Assis Boldt** (https://fboldt.github.io/). 

Cada notebook contém implementações desde zero (*from scratch*) de algoritmos fundamentais, acompanhadas de explicações pedagógicas detalhadas sobre os conceitos, intuição matemática e aplicações práticas.

---

## 📚 Estrutura do Curso

### Fase 1️⃣: Regressão & Fundamentos
Aprenda os conceitos básicos de aprendizado supervisionado através de modelos contínuos.

| Aula | Tópico | Objetivos | Datasets |
|:----:|--------|-----------|----------|
| **01a** | Regressão Linear | Carregar dados, exploração EDA, implementar modelo linear do zero, métricas MSE/MAE/RMSE | Diabetes (sklearn) |
| **02a** | Regressão Avançada | Análise aprofundada de modelos de regressão | Diabetes (sklearn) |
| **02b** | Equação Normal | Resolver regressão linear via álgebra linear (pseudoinversa) | Diabetes (sklearn) |
| **02c** | Comparação de Regressores | Comparar LinearRegression, Ridge, Lasso, SVM | Diabetes (sklearn) |

---

### Fase 2️⃣: Classificação & Validação
Transição para problemas discretos e técnicas de avaliação rigorosa.

| Aula | Tópico | Objetivos | Datasets |
|:----:|--------|-----------|----------|
| **03a** | Classificação Linear | Usar regressão para classificação, introduzir limiar (*threshold*), análise de correlações | Breast Cancer (sklearn) |
| **03b** | Normalização & Scaling | Padronizar features (StandardScaler, MinMaxScaler) | Breast Cancer (sklearn) |
| **04a** | K-Nearest Neighbors (KNN) | Implementar KNN, entender vizinhança, hiperparâmetro k | Iris (sklearn) |
| **04b** | Pipelines | Encadear preprocessamento + modelo para treino/teste reproduzível | Multiple |
| **04c** | Validação Cruzada | Evitar vazamento de dados, usar k-fold cross-validation | Multiple |
| **04d** | Grid Search | Otimização automática de hiperparâmetros via busca em grade | Multiple |

---

### Fase 3️⃣: Árvores & Ensemble
Modelos baseados em árvores e técnicas de ensemble.

| Aula | Tópico | Objetivos | Datasets |
|:----:|--------|-----------|----------|
| **05** | Árvores de Decisão (Discreto) | Construir árvores para atributos discretos, impureza (Gini/Entropia) | Custom |
| **05b** | Árvores de Decisão (Contínuo) | Estender para atributos contínuos, limiares de split | Multiple |
| **06a** | Ensembles | Random Forest, Gradient Boosting, votação e stacking | Multiple |
| **07a** | Projeto: Titanic | Aplicação prática: exploração, limpeza, feature engineering, modelagem | Titanic |

---

### Fase 4️⃣: Aprendizado Não Supervisionado
Agrupamento e redução de dimensionalidade sem rótulos.

| Aula | Tópico | Objetivos | Datasets |
|:----:|--------|-----------|----------|
| **08a** | K-Means (From Scratch) | Implementar K-Means do zero, inicialização, convergência, visualização | Iris (sklearn) |
| **08b** | Método do Cotovelo | Encontrar k ótimo através do gráfico de inércia | Iris (sklearn) |
| **08c** | K-Means para Feature Selection | Usar clustering como técnica de redução/seleção de features | Iris (sklearn) |
| **09a** | DBSCAN | Clustering baseado em densidade, parâmetros eps e min_samples | Synthetic |
| **09b** | Semi-Supervised Learning | Combinar dados rotulados e não rotulados | Mixed |

---

## 🚀 Como Usar Este Repositório

### Pré-requisitos
- Python 3.8+
- Jupyter Notebook ou Google Colab
- Bibliotecas: `scikit-learn`, `pandas`, `numpy`, `matplotlib`

### Instalação de Dependências
```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```

### Executando as Aulas

#### Localmente (VS Code / Jupyter Lab)
```bash
jupyter notebook machine-learning/aula01a_regression.ipynb
```

#### No Google Colab
Cada notebook possui um botão "Open in Colab" no topo para execução em nuvem sem necessidade de instalação local.

---

## 💡 Principais Conceitos Cobertos

### Supervised Learning (Supervisionado)
- ✅ Regressão: Linear, Polinomial, Ridge, Lasso
- ✅ Classificação: Logística, KNN, Árvores, Ensembles
- ✅ Validação: Train-Test Split, Cross-Validation, Métricas

### Unsupervised Learning (Não Supervisionado)
- ✅ Clustering: K-Means, DBSCAN
- ✅ Redução de Dimensionalidade: Feature Selection
- ✅ Semi-Supervised Learning

### Técnicas Fundamentais
- ✅ Exploração de Dados (EDA)
- ✅ Normalização e Scaling de Features
- ✅ Seleção de Hiperparâmetros (Grid Search)
- ✅ Pipelines de ML
- ✅ Avaliação de Modelos

---

## 📊 Datasets Utilizados

| Dataset | Origem | Tipo | Amostras | Features |
|---------|--------|------|----------|----------|
| **Diabetes** | sklearn | Regressão | 442 | 10 |
| **Breast Cancer** | sklearn | Classificação Binária | 569 | 30 |
| **Iris** | sklearn | Classificação Multiclasse | 150 | 4 |
| **Titanic** | Kaggle | Classificação (Projeto) | 891 | 11 |

---

## 📖 Abordagem Pedagógica

Cada notebook segue uma estrutura didática consistente:

1. **Carregamento e Exploração (EDA):** Entender os dados antes de modelar
2. **Implementação do Zero:** Construir algoritmos manualmente para entender a matemática
3. **Uso da Biblioteca:** Aplicar sklearn para eficiência
4. **Visualização:** Gráficos para intuição geométrica
5. **Métricas e Avaliação:** Medir desempenho de forma rigorosa
6. **Discussão Pedagógica:** Explicações em markdown sobre "O quê", "Por quê" e "Como"

> **Filosofia do Curso:** *Aprender Machine Learning é entender a matemática através da prática e visualização, não apenas usar bibliotecas.*

---

## 🔗 Referências e Recursos

- **Professor:** [Francisco de Assis Boldt](https://fboldt.github.io/)
- **Documentação sklearn:** https://scikit-learn.org/stable/
- **NumPy Reference:** https://numpy.org/doc/
- **Matplotlib Gallery:** https://matplotlib.org/gallery.html

---

## 📝 Licença e Atribuição

> [!IMPORTANT]
> Este repositório contém materiais educacionais baseados nas aulas ministradas pelo **Prof. Francisco de Assis Boldt**. Todos os créditos, direitos autorais e propriedade intelectual dos conceitos e códigos base pertencem ao professor.
>
> Estes notebooks são fornecidos para fins educacionais e de estudo pessoal no contexto do programa de Mestrado em Computação Aplicada.
