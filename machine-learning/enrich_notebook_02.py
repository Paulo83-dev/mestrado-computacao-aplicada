import json

file_path = '/home/paulo/.gemini/antigravity/scratch/mestrado-computacao-aplicada/machine-learning/aula02a_regression.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# DEFINIÇÃO DOS NOVOS BLOCOS DE MARKDOWN DE INTRODUÇÃO E EXPLICAÇÃO PEDAGÓGICA

md_intro = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🧠 Aula 02a: Regressão Linear, Validação e Estimadores Scikit-Learn\n",
        "\n",
        "Nesta aula, expandimos o conceito de **Regressão Linear** visto anteriormente. Vamos abordar:\n",
        "1. **Validação de Modelos:** A divisão dos dados em conjuntos de Treino e Teste.\n",
        "2. **Análise Gráfica dos Erros:** Como o erro varia em função dos parâmetros ($a$ e $b$) e por que isso nos ajuda a encontrar a melhor reta.\n",
        "3. **Modelo de Baseline:** O que é o modelo da média e para que serve.\n",
        "4. **Métricas de Qualidade:** O coeficiente de determinação ($R^2$).\n",
        "5. **Estrutura Scikit-Learn:** Como criar nossos próprios estimadores seguindo a API padrão do `sklearn` (`BaseEstimator` e `RegressorMixin`).\n",
        "6. **Gradiente Descendente:** A implementação manual de um algoritmo de otimização que ajusta os pesos da reta automaticamente!"
    ]
}

md_split = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### ✂️ Divisão em Treino e Teste\n",
        "\n",
        "#### 🔍 Por que dividimos os dados?\n",
        "Se usarmos todos os dados para treinar o modelo e depois avaliarmos o erro nesses mesmos dados, teremos uma falsa impressão de que o modelo é excelente. Em machine learning, queremos avaliar o poder de **generalização** do modelo, ou seja, sua capacidade de acertar previsões para novos dados que ele nunca viu antes.\n",
        "\n",
        "Por isso, separamos o dataset em:\n",
        "- **Treino (Train):** Usado para ajustar os parâmetros do modelo (ex: descobrir a inclinação $a$ e viés $b$). Geralmente representa 80% do total.\n",
        "- **Teste (Test):** Guardado a sete chaves e usado apenas para calcular o erro final. Funciona como a prova final do modelo. Geralmente representa 20% do total."
    ]
}

md_error_surface = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📉 O Espaço de Parâmetros e Superfícies de Erro\n",
        "\n",
        "#### 🔍 O que a função `plot_metric_by_a` nos mostra?\n",
        "Esta função plota no eixo horizontal diferentes valores de inclinação $a$ (para a reta $y = ax + b$) e, no eixo vertical, o erro resultante (MAE, MSE ou RMSE) correspondente a cada $a$.\n",
        "\n",
        "#### 🎯 Qual a intenção pedagógica?\n",
        "1. **Convexidade:** Note que o gráfico resultante tem um formato em **U** (uma parábola). Isso significa que existe um único valor ideal de $a$ que minimiza o erro (o ponto mais baixo da curva).\n",
        "2. **Visualização do Mínimo Global:** Em vez de chutar valores aleatórios de $a$, o gráfico nos mostra visualmente que a busca pela melhor reta é o equivalente matemático a \"descer a ladeira\" até encontrar o vale da curva."
    ]
}

md_baseline = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📊 Modelo de Baseline (Modelo da Média)\n",
        "\n",
        "#### 🔍 O que é e para que serve?\n",
        "O **Modelo da Média** (`modelo_media`) é o preditor mais simples possível: ele ignora totalmente os exames do paciente ($X$) e simplesmente chuta que todo paciente terá um valor de progressão de diabetes igual à **média histórica** de todos os pacientes do conjunto de treino.\n",
        "\n",
        "#### 🎯 Qual a intenção pedagógica?\n",
        "Ele funciona como o nosso **ponto de partida ou benchmark (baseline)**. Qualquer modelo inteligente que criarmos (como a Regressão Linear) **obrigatoriamente deve ter um erro menor que o da média**. Se o nosso modelo não superar a média, significa que ele não aprendeu absolutamente nada útil a partir dos dados."
    ]
}

md_r2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 🏆 Coeficiente de Determinação ($R^2$ Score)\n",
        "\n",
        "#### 🔍 O que é o $R^2$?\n",
        "Enquanto o MSE ou MAE nos dão o erro nas unidades originais dos dados (o que pode ser difícil de avaliar se não conhecermos a escala), o $R^2$ é uma métrica **normalizada** que mede a proporção da variação dos dados que nosso modelo consegue explicar.\n",
        "\n",
        "A fórmula matemática é:\n",
        "$$R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}$$\n",
        "Onde:\n",
        "- $SS_{res}$ é a soma dos erros quadráticos do nosso modelo (MSE).\n",
        "- $SS_{tot}$ é a soma dos erros quadráticos do modelo da média (variância total dos dados).\n",
        "\n",
        "#### 💡 Interpretando o $R^2$:\n",
        "- **$R^2 = 1$:** Modelo perfeito (zero erro).\n",
        "- **$R^2 = 0$:** O modelo é tão bom quanto simplesmente chutar a média para todo mundo.\n",
        "- **$R^2 < 0$:** O modelo consegue ser pior do que apenas chutar a média!"
    ]
}

md_sklearn_api = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 🔌 Integrando com a API do Scikit-Learn\n",
        "\n",
        "#### 🔍 Por que estender `BaseEstimator` e `RegressorMixin`?\n",
        "O ecossistema do Python para ciência de dados é muito forte porque as ferramentas conversam entre si. Criar funções soltas funciona no início, mas para usar ferramentas automáticas (como Cross-Validation ou GridSearch), precisamos que o nosso modelo siga o padrão de projeto do Scikit-Learn.\n",
        "\n",
        "Ao herdar de `BaseEstimator` e `RegressorMixin`, nossa classe ganha automaticamente métodos importantes (como `.score()`) e pode ser tratada como qualquer outro algoritmo oficial do Scikit-Learn.\n",
        "\n",
        "#### 🛠️ O Contrato da API:\n",
        "1. **`fit(X, y)`:** Método de treinamento. Aqui calculamos os parâmetros (como a média no `AverageRegressor` ou os pesos no `LinearRegressor`).\n",
        "2. **`predict(X)`:** Método de previsão. Recebe novas características $X$ e devolve as previsões estimadas."
    ]
}

md_gradient_descent = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### ⚡ Gradiente Descendente (Gradient Descent)\n",
        "\n",
        "#### 🔍 O que é o Gradiente Descendente?\n",
        "Nos exemplos anteriores, estávamos chutando os valores de $a$ e $b$ na mão ou buscando graficamente. O **Gradiente Descendente** é o algoritmo de otimização que faz essa busca de forma matemática e automática.\n",
        "\n",
        "Ele funciona assim:\n",
        "1. Começa com coeficientes (`coefs_`) e viés (`intercept_`) aleatórios.\n",
        "2. Calcula a previsão (`y_pred`) e o erro (`error = y - y_pred`).\n",
        "3. Calcula a direção em que o erro diminui (o gradiente) e dá um pequeno passo nessa direção. O tamanho do passo é controlado pelo **`learning_rate`** (taxa de aprendizado).\n",
        "4. Repete esse processo por **`max_iter`** vezes.\n",
        "\n",
        "#### 💡 Intuição dos Hiperparâmetros:\n",
        "- **`learning_rate` (Taxa de Aprendizado):** Se for muito grande, o algoritmo dá passos gigantescos e pode passar direto pelo mínimo (divergir). Se for pequeno demais, ele dá passos microscópicos e demora muito para treinar.\n",
        "- **`max_iter` (Épocas/Iterações):** Quantas vezes o algoritmo vai olhar para os dados e atualizar os pesos."
    ]
}

md_multiple_regression = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 🌌 Regressão Linear Múltipla\n",
        "\n",
        "#### 🔍 O que mudou?\n",
        "Até agora, estávamos usando apenas **uma única característica** (o IMC do paciente, na variável `X2`) para tentar adivinhar a progressão da diabetes.\n",
        "\n",
        "Neste último bloco, passamos o conjunto de treino completo (`X_train`), que contém **todas as 10 características** (idade, sexo, IMC, pressão arterial, e seis exames de sangue diferente).\n",
        "\n",
        "#### 🎯 Intenção Pedagógica:\n",
        "1. O modelo agora ajusta 10 coeficientes (um para cada coluna) e 1 intercepto.\n",
        "2. Observe como o erro MSE cai significativamente. Isso demonstra que ao combinar múltiplas fontes de informação o modelo se torna muito mais preciso e capaz do que usando apenas uma variável isolada."
    ]
}

new_cells = []

# Variável de controle para evitar duplicar as inserções se rodar o script de novo
keywords_to_check = ["Regressão Linear, Validação", "Divisão em Treino e Teste", "Espaço de Parâmetros", "Modelo de Baseline", "r2_score", "Integrando com a API", "Gradiente Descendente", "Regressão Linear Múltipla"]

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        src_str = "".join(cell.get('source', []))
        if any(kw in src_str for kw in keywords_to_check):
            continue # Evita duplicados
        new_cells.append(cell)
        continue

    src = "".join(cell['source'])

    if "load_diabetes()" in src:
        # Colab Badge / Intro
        new_cells.append(md_intro)
        
        # Código 1 comentado
        cell['source'] = [
            "# 1. Importa a função do scikit-learn para carregar o dataset de diabetes\n",
            "from sklearn.datasets import load_diabetes\n",
            "\n",
            "# 2. Carrega o conjunto de dados em uma variável dicionário chamada 'data'\n",
            "data = load_diabetes()\n",
            "\n",
            "# 3. Exibe as chaves principais do dicionário para conhecermos a estrutura do dataset\n",
            "print(data.keys())"
        ]
        new_cells.append(cell)

    elif "X = data.data" in src:
        cell['source'] = [
            "# Extrai a matriz de características (os dados dos exames) e exibe suas dimensões (linhas, colunas)\n",
            "X = data.data\n",
            "print(X.shape)"
        ]
        new_cells.append(cell)

    elif "y = data.target" in src:
        cell['source'] = [
            "# Extrai o vetor alvo (a progressão da diabetes) e exibe suas dimensões (vetor unidimensional)\n",
            "y = data.target\n",
            "print(y.shape)"
        ]
        new_cells.append(cell)

    elif "import matplotlib.pyplot as plt\nfor i in [2]:" in src:
        cell['source'] = [
            "# Importa biblioteca gráfica e exibe a relação entre a coluna 2 (IMC) e o alvo (diabetes)\n",
            "import matplotlib.pyplot as plt\n",
            "for i in [2]: #range(X.shape[1]):\n",
            "    print(i)\n",
            "    plt.scatter(X[:,i],y)\n",
            "    plt.xlabel(data.feature_names[i])\n",
            "    plt.ylabel(\"target\")\n",
            "    plt.show()"
        ]
        new_cells.append(cell)

    elif "def train_test_split" in src:
        # Inserir markdown explicativo sobre treino/teste antes do código
        new_cells.append(md_split)
        
        # Código split comentado
        cell['source'] = [
            "import numpy as np\n",
            "\n",
            "# Função personalizada para dividir o conjunto de dados em treino (80%) e teste (20%)\n",
            "def train_test_split(X, y, test_size=0.2, random_state=42):\n",
            "  # 1. Garante a reprodutibilidade dos números aleatórios\n",
            "  np.random.seed(random_state)\n",
            "  \n",
            "  # 2. Embaralha os índices do dataset para não pegar dados ordenados de forma tendenciosa\n",
            "  idx = np.random.permutation(len(X))\n",
            "  X = X[idx]\n",
            "  y = y[idx]\n",
            "  \n",
            "  # 3. Calcula o ponto exato de corte (ex: se 80% treino, corta em 80% do tamanho total)\n",
            "  split_idx = int(len(X)*(1-test_size))\n",
            "  \n",
            "  # 4. Divide as matrizes de características e vetores alvo com base no corte calculado\n",
            "  X_train, X_test = X[:split_idx], X[split_idx:]\n",
            "  y_train, y_test = y[:split_idx], y[split_idx:]\n",
            "  \n",
            "  return X_train, X_test, y_train, y_test\n",
            "\n",
            "# Executamos a divisão e exibimos as dimensões de cada um dos novos quatro conjuntos\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)"
        ]
        new_cells.append(cell)

    elif "X2 = X_train[:,2:3]" in src:
        cell['source'] = [
            "# Extraímos apenas os dados de IMC (coluna de índice 2) do conjunto de TREINO\n",
            "# Mantemos o formato bidimensional usando o fatiamento 2:3 para que o shape seja (N, 1)\n",
            "X2 = X_train[:,2:3]\n",
            "print(X2.shape)"
        ]
        new_cells.append(cell)

    elif "def modelo_linear(X, a, b):" in src:
        cell['source'] = [
            "# Função matemática da reta preditora: y_pred = a * X + b\n",
            "def modelo_linear(X, a, b):\n",
            "  y = a*X + b\n",
            "  return y"
        ]
        new_cells.append(cell)

    elif "def plot_results(X, y, ypred):" in src:
        cell['source'] = [
            "# Função auxiliar para exibir as bolinhas reais em azul (com transparência) e a reta preditora em vermelho\n",
            "def plot_results(X, y, ypred):\n",
            "  plt.scatter(X, y, alpha=0.3)\n",
            "  plt.plot(X, ypred, 'r')\n",
            "  plt.show()"
        ]
        new_cells.append(cell)

    elif "y_pred = modelo_linear(X2, 1000, 100)" in src:
        cell['source'] = [
            "# Chute 1: a=1000, b=100. Vamos ver como a reta se comporta nos dados de treino.\n",
            "y_pred = modelo_linear(X2, 1000, 100)\n",
            "plot_results(X2, y_train, y_pred)"
        ]
        new_cells.append(cell)

    elif "y_pred = modelo_linear(X2, 1500, 150)" in src:
        cell['source'] = [
            "# Chute 2: a=1500, b=150. Um pouco mais inclinado e mais alto.\n",
            "y_pred = modelo_linear(X2, 1500, 150)\n",
            "plot_results(X2, y_train, y_pred)"
        ]
        new_cells.append(cell)

    elif "def mae(y, y_pred):" in src:
        cell['source'] = [
            "# Função para calcular o Erro Médio Absoluto (MAE) manual\n",
            "def mae(y, y_pred):\n",
            "  # Redimensiona y_pred para 1D se necessário, para garantir compatibilidade na subtração\n",
            "  diff = y - y_pred.reshape(len(y), )\n",
            "  abs_diff = abs(diff)\n",
            "  sum_error = np.sum(abs_diff)\n",
            "  mae = sum_error / len(y)\n",
            "  return mae\n",
            "\n",
            "# Calcula e compara o MAE dos dois chutes anteriores no treino\n",
            "mae(y_train, modelo_linear(X2, 1000, 100)), mae(y_train, modelo_linear(X2, 1500, 150))"
        ]
        new_cells.append(cell)

    elif "def plot_metric_by_a" in src:
        # Adiciona a explicação da superfície de erro antes do código de plotagem
        new_cells.append(md_error_surface)
        
        cell['source'] = [
            "# Função que desenha a curva de erro correspondente a diferentes valores de inclinação 'a'\n",
            "def plot_metric_by_a(X, y, metric, inference_function, min_a, max_a):\n",
            "  # Cria 50 valores igualmente espaçados no intervalo de a de interesse\n",
            "  a_values = np.linspace(min_a, max_a)\n",
            "  # Realiza as inferências (o broadcast gera as predições para cada valor de a)\n",
            "  y_preds = inference_function(X, a_values, 0)\n",
            "  # Calcula o erro usando a métrica passada para cada uma das predições realizadas\n",
            "  metric_values = [metric(y, y_pred) for y_pred in y_preds.T]\n",
            "  # Plota os erros em função de a\n",
            "  plt.plot(a_values, metric_values)\n",
            "  plt.xlabel(\"a\")\n",
            "  plt.ylabel(metric.__name__)\n",
            "\n",
            "# Plota o comportamento do MAE variando o 'a' entre 300 e 1500\n",
            "plot_metric_by_a(X2, y_train, mae, modelo_linear, 300, 1500)\n",
            "plt.show()"
        ]
        new_cells.append(cell)

    elif "def mse(y, y_pred):" in src:
        cell['source'] = [
            "# Função para calcular o Erro Quadrático Médio (MSE)\n",
            "def mse(y, y_pred):\n",
            "  diff = y - y_pred.reshape(len(y), )\n",
            "  squared = diff**2\n",
            "  mean = sum(squared)/len(squared)\n",
            "  return mean\n",
            "\n",
            "# Calcula e compara o MSE dos dois chutes de treino\n",
            "mse(y_train, modelo_linear(X2, 1000, 100)), mse(y_train, modelo_linear(X2, 1500, 150))"
        ]
        new_cells.append(cell)

    elif "plot_metric_by_a(X2, y_train, mse, modelo_linear, 400, 1500)" in src:
        cell['source'] = [
            "# Plota como o MSE varia ao alterar o coeficiente 'a' de 400 a 1500\n",
            "plot_metric_by_a(X2, y_train, mse, modelo_linear, 400, 1500)\n",
            "plt.show()"
        ]
        new_cells.append(cell)

    elif "def rmse(y, ypred):" in src:
        cell['source'] = [
            "# Função para calcular a raiz do MSE (RMSE)\n",
            "def rmse(y, ypred):\n",
            "  return np.sqrt(mse(y, ypred))\n",
            "\n",
            "# Calcula e compara o RMSE dos dois chutes iniciais\n",
            "rmse(y_train, modelo_linear(X2, 1000, 100)), rmse(y_train, modelo_linear(X2, 1500, 150))"
        ]
        new_cells.append(cell)

    elif "plot_metric_by_a(X2, y_train, rmse, modelo_linear, 400, 1500)" in src:
        cell['source'] = [
            "# Plota o comportamento do RMSE variando o 'a' entre 400 e 1500\n",
            "plot_metric_by_a(X2, y_train, rmse, modelo_linear, 400, 1500)\n",
            "plt.show()"
        ]
        new_cells.append(cell)

    elif "def modelo_media(y):" in src:
        # Insere a explicação do Baseline da Média
        new_cells.append(md_baseline)
        
        cell['source'] = [
            "# Função do modelo da média: simplesmente prevê a média do alvo para todas as observações\n",
            "def modelo_media(y):\n",
            "  media = np.mean(y)\n",
            "  # Cria um array preenchido com 1s no formato de y e multiplica pelo valor médio\n",
            "  y_pred = np.ones_like(y)*media\n",
            "  return y_pred\n",
            "\n",
            "# Comparamos o MSE do nosso chute manual (a=1500, b=150) com o MSE do modelo da média\n",
            "mse(y_train, modelo_linear(X2, 1500, 150)), mse(y_train, modelo_media(y_train))"
        ]
        new_cells.append(cell)

    elif "y_pred = modelo_media(y_train)\nplot_results" in src:
        cell['source'] = [
            "# Gera as predições de linha média e plota os resultados graficamente (uma linha horizontal perfeita)\n",
            "y_pred = modelo_media(y_train)\n",
            "plot_results(X2, y_train, y_pred)"
        ]
        new_cells.append(cell)

    elif "def r2_score(y, y_pred):" in src:
        # Insere a explicação do R2
        new_cells.append(md_r2)
        
        cell['source'] = [
            "# Função que calcula o R² Score a partir do erro do modelo e do erro da média (total)\n",
            "def r2_score(y, y_pred):\n",
            "  ss_res = mse(y, y_pred)\n",
            "  ss_tot = mse(y, modelo_media(y))\n",
            "  r2 = 1 - (ss_res/ss_tot)\n",
            "  return r2\n",
            "\n",
            "# Calcula o R² para a nossa reta linear e para o próprio modelo da média (que deve ser exatamente 0.0)\n",
            "r2_score(y_train, modelo_linear(X2, 1500, 150)), r2_score(y_train, modelo_media(y_train))"
        ]
        new_cells.append(cell)

    elif "class AverageRegressor" in src:
        # Insere a explicação da API do sklearn
        new_cells.append(md_sklearn_api)
        
        cell['source'] = [
            "from sklearn.base import BaseEstimator, RegressorMixin\n",
            "\n",
            "# Criação do estimador padrão da média seguindo a API do Scikit-Learn\n",
            "class AverageRegressor(BaseEstimator, RegressorMixin):\n",
            "  def fit(self, X, y):\n",
            "    # No fit, aprendemos a média das etiquetas de treino e guardamos no atributo interno 'media_'\n",
            "    self.media_ = np.mean(y)\n",
            "    return self\n",
            "    \n",
            "  def predict(self, X):\n",
            "    # No predict, criamos um vetor com a média aprendida para cada amostra do conjunto X fornecido\n",
            "    y_pred = np.ones(shape=(X.shape[0],))*self.media_\n",
            "    return y_pred\n",
            "\n",
            "# Instancia o regressor, treina e exibe o MSE correspondente aos dados de treino\n",
            "regressor = AverageRegressor()\n",
            "regressor.fit(X2, y_train)\n",
            "y_pred = regressor.predict(X2)\n",
            "print(mse(y_train, y_pred))\n",
            "plot_results(X2, y_train, y_pred)"
        ]
        new_cells.append(cell)

    elif "class LinearRegressor" in src and "self.a_ = a" in src:
        cell['source'] = [
            "# Estimador linear com pesos fixos passados na inicialização\n",
            "class LinearRegressor(BaseEstimator, RegressorMixin):\n",
            "  def __init__(self, a, b):\n",
            "    self.a_ = a\n",
            "    self.b_ = b\n",
            "    \n",
            "  def fit(self, X, y):\n",
            "    # Não há treinamento real aqui pois os parâmetros foram pré-definidos\n",
            "    return self\n",
            "    \n",
            "  def predict(self, X):\n",
            "    # Realiza a predição y = a*X + b, redimensionando o vetor de saída para 1D\n",
            "    y_pred = self.a_*X + self.b_\n",
            "    return y_pred.reshape(X.shape[0], )\n",
            "\n",
            "# Executa o estimador para os parâmetros a=1500 e b=150\n",
            "regressor = LinearRegressor(1500, 150)\n",
            "regressor.fit(X2, y_train)\n",
            "y_pred = regressor.predict(X2)\n",
            "print(mse(y_train, y_pred))\n",
            "plot_results(X2, y_train, y_pred)"
        ]
        new_cells.append(cell)

    elif "class LinearRegressor" in src and "np.random.rand()" in src:
        cell['source'] = [
            "# Estimador linear que no 'fit' simplesmente chuta valores aleatórios para os pesos\n",
            "class LinearRegressor(BaseEstimator, RegressorMixin):\n",
            "  def fit(self, X, y):\n",
            "    self.a_ = np.random.rand()\n",
            "    self.b_ = np.random.rand()\n",
            "    return self\n",
            "    \n",
            "  def predict(self, X):\n",
            "    y_pred = self.a_*X + self.b_\n",
            "    return y_pred.reshape(X.shape[0], )\n",
            "\n",
            "# Executa e exibe o comportamento da reta aleatória resultante\n",
            "regressor = LinearRegressor()\n",
            "regressor.fit(X2, y_train)\n",
            "y_pred = regressor.predict(X2)\n",
            "print(mse(y_train, y_pred))\n",
            "plot_results(X2, y_train, y_pred)"
        ]
        new_cells.append(cell)

    elif "class LinearRegressor" in src and "max_iter" in src:
        # Adiciona explicação de Gradiente Descendente
        new_cells.append(md_gradient_descent)
        
        cell['source'] = [
            "# Estimador linear com otimização manual por Gradiente Descendente\n",
            "class LinearRegressor(BaseEstimator, RegressorMixin):\n",
            "  def __init__(self, max_iter=1000, learning_rate=0.001):\n",
            "    self.max_iter = max_iter\n",
            "    self.learning_rate = learning_rate\n",
            "\n",
            "  def fit(self, X, y):\n",
            "    # 1. Inicializa pesos (coefs_) e viés (intercept_) de forma aleatória\n",
            "    self.coefs_ = np.random.rand(X.shape[1])\n",
            "    self.intercept_ = np.random.rand()\n",
            "    \n",
            "    # 2. Loop de otimização de parâmetros\n",
            "    for i in range(self.max_iter):\n",
            "      # Previsão com os pesos atuais\n",
            "      y_pred = self.predict(X)\n",
            "      # Erro ponto a ponto (Real - Previsto)\n",
            "      error = y - y_pred\n",
            "      \n",
            "      # Atualiza coeficientes aproximando do gradiente da função de erro\n",
            "      self.coefs_ += X.T @ error * self.learning_rate\n",
            "      # Atualiza o intercepto com base na soma total do erro\n",
            "      self.intercept_ += error.sum() * self.learning_rate\n",
            "    return self\n",
            "\n",
            "  def predict(self, X):\n",
            "    # Previsão geral: X @ pesos + intercepto\n",
            "    y_pred = X @ self.coefs_ + self.intercept_\n",
            "    return y_pred.reshape(X.shape[0], )\n",
            "\n",
            "# Otimiza os pesos usando o Gradiente Descendente no IMC de treino\n",
            "regressor = LinearRegressor()\n",
            "regressor.fit(X2, y_train)\n",
            "y_pred = regressor.predict(X2)\n",
            "print(\"Pesos finais aprendidos (IMC):\", regressor.coefs_, regressor.intercept_)\n",
            "print(\"MSE final do treino:\", mse(y_train, y_pred))\n",
            "plot_results(X2, y_train, y_pred)"
        ]
        new_cells.append(cell)

    elif "y_pred = modelo_linear(X2, 529.64525529, 154.8226859117432)" in src:
        cell['source'] = [
            "# Verificando o erro utilizando a reta ideal encontrada analiticamente pelo professor\n",
            "y_pred = modelo_linear(X2, 529.64525529, 154.8226859117432)\n",
            "print(mse(y_train, y_pred))"
        ]
        new_cells.append(cell)

    elif "regressor.fit(X_train, y_train)" in src:
        # Insere explicação da Regressão Múltipla antes do código correspondente
        new_cells.append(md_multiple_regression)
        
        cell['source'] = [
            "# Treinando o modelo de Gradiente Descendente utilizando TODOS os exames (X_train completo)\n",
            "regressor = LinearRegressor()\n",
            "regressor.fit(X_train, y_train)\n",
            "y_pred = regressor.predict(X_train)\n",
            "\n",
            "# Exibe os pesos aprendidos para cada um dos 10 atributos de exames e o intercepto final\n",
            "print(\"10 Pesos Aprendidos:\", regressor.coefs_, \"\\nIntercepto:\", regressor.intercept_)\n",
            "print(\"MSE com todas as características:\", mse(y_train, y_pred))\n",
            "\n",
            "# Plota graficamente a comparação do valor real com a nossa previsão (usando o IMC como referência do eixo X)\n",
            "plt.scatter(X2, y_train, alpha=0.3, label='Real (IMC vs Target)')\n",
            "plt.scatter(X2, y_pred, alpha=0.3, color='r', label='Previsão (Todas as Features)')\n",
            "plt.legend()\n",
            "plt.show()"
        ]
        new_cells.append(cell)

    else:
        new_cells.append(cell)

# Salva o arquivo modificado de volta em aula02a_regression.ipynb
nb['cells'] = new_cells

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook aula02a_regression.ipynb modified successfully with rich explanations and comments!")
