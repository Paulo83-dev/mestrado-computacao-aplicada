# 🚀 Quick Start - Machine Learning

Comece a estudar Machine Learning em 5 minutos!

---

## 1️⃣ Instalação (2 minutos)

### Com pip (Recomendado)
```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```

### Com Anaconda
```bash
conda create -n ml-mestrado python=3.10
conda activate ml-mestrado
conda install scikit-learn pandas numpy matplotlib seaborn jupyter
```

### No Google Colab (Sem instalação!)
Não precisa instalar nada. Abra qualquer notebook e clique no botão "Open in Colab" no topo. Pronto! ☁️

---

## 2️⃣ Executar as Aulas (1 minuto)

### Localmente
```bash
# Navegue até a pasta
cd machine-learning

# Abra o Jupyter
jupyter notebook

# Clique na aula que quer estudar
# Exemplo: aula01a_regression.ipynb
```

### No Google Colab
Abra este link direto no navegador:
- [Aula 01a - Regressão Linear](https://colab.research.google.com/github/fboldt/aulasml/blob/master/aula01a_regression.ipynb)
- [Aula 03a - Classificação](https://colab.research.google.com/github/fboldt/aulasml/blob/master/aula03a_classifica%C3%A7%C3%A3o.ipynb)
- [Aula 08a - K-Means](https://colab.research.google.com/github/fboldt/aulasml/blob/master/aula08a_kmeans.ipynb)

---

## 3️⃣ Primeiro Contato (2 minutos)

Abra a **Aula 01a** (`aula01a_regression.ipynb`) e execute cada célula de cima para baixo clicando:
- 🖱️ **Shift + Enter** (Jupyter local)
- ▶️ **Ctrl + Enter** (Google Colab)

Você verá:
1. Carregamento do dataset Diabetes
2. Criação de um gráfico de dispersão
3. Implementação manual de regressão linear
4. Cálculo de métricas de erro

---

## 📖 Sugestão de Ordem de Estudo

### Se você é iniciante em ML:
```
1. Aula 01a → Aula 02a → Aula 02c
   (Regressão Linear, Equação Normal, Comparação de Modelos)

2. Aula 03a → Aula 03b
   (Classificação Linear, Normalização)

3. Aula 04a → Aula 04c → Aula 04d
   (KNN, Cross-Validation, Grid Search)

4. Aula 07a (Projeto Titanic - Aplicação Prática)

5. Aula 08a → Aula 09a (Clustering)
```

### Se você já sabe o básico:
```
Pule direto para:
- Aula 06a (Ensembles - Random Forest, Gradient Boosting)
- Aula 07a (Projeto Titanic - Desafio Real)
- Aula 09b (Semi-Supervised Learning)
```

---

## ❓ Dúvidas Comuns

### "E se eu não conseguir instalar as dependências?"
→ Use Google Colab! Zero instalação, 100% online, com GPU gratuita.

### "Qual a ordem correta para assistir as aulas?"
→ Siga a sugestão acima. Cada aula constrói sobre a anterior.

### "Posso fazer isto sem conhecimento prévio de Python?"
→ É recomendável saber Python básico. Se precisar revisar, veja: [Python para Iniciantes](https://www.python.org/about/gettingstarted/)

### "Onde vejo a solução dos exercícios?"
→ Cada notebook é autoexplicativo. Leia as células de markdown para entender o "por quê" de cada passo.

---

## 🎯 Seu Primeiro Experimento

Após estudar a Aula 01a, tente:

```python
from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt

# Carregue o dataset
data = load_diabetes()
X = data.data[:, 2:3]  # Escolha uma feature
y = data.target

# Crie um scatter plot
plt.scatter(X, y, alpha=0.5)
plt.xlabel(data.feature_names[2])
plt.ylabel('Progressão da Diabetes')
plt.show()

# Calcule a correlação
correlation = np.corrcoef(X.flatten(), y)[0, 1]
print(f"Correlação: {correlation:.3f}")
```

Parabéns! Você acabou de fazer Análise Exploratória de Dados (EDA)! 🎉

---

## 🔧 Verificação de Instalação

Execute isto em uma célula Jupyter para confirmar que tudo está instalado:

```python
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(f"✅ scikit-learn: {sklearn.__version__}")
print(f"✅ pandas: {pd.__version__}")
print(f"✅ NumPy: {np.__version__}")
print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
print("\n🎉 Tudo pronto para aprender ML!")
```

---

## 📚 Próximos Passos

Depois de completar todas as aulas:

1. **Projetos Pessoais:** Aplique ML em dados que você ama
2. **Kaggle Competitions:** Desafios de ML com prêmios reais
3. **Deep Learning:** Entre em redes neurais com TensorFlow/PyTorch
4. **Research Papers:** Leia artigos clássicos em arXiv

---

## 🤝 Precisa de Ajuda?

- 📧 Entre em contato com seu professor
- 🔗 Consulte a [documentação do scikit-learn](https://scikit-learn.org/stable/)
- 📺 Assista a [vídeos 3Blue1Brown sobre Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

---

**Pronto para começar?** 🚀

→ Abra [`machine-learning/aula01a_regression.ipynb`](./machine-learning/aula01a_regression.ipynb) e execute a primeira célula!
