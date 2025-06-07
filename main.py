import matplotlib
matplotlib.use('Agg')  # Usar backend sem GUI (evita erro do Tkinter)

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import string

# Baixar recursos necessários (somente na primeira execução)
nltk.download('stopwords')
nltk.download('rslp')

# 1. Documentos de exemplo (solicitações para um service desk)
docs = [
    "Gostaria de saber como mudar minha senha.",
    "Não estou conseguindo imprimir os relatórios da semana.",
    "O sistema está fora do ar e preciso trabalhar.",
    "Erro ao tentar salvar os dados no sistema financeiro.",
    "Como faço para reinstalar o aplicativo de RH?",
    "A tela está travando quando tento fazer login.",
    "Quero solicitar acesso ao sistema de chamados.",
    "Minha conta foi bloqueada após várias tentativas de login.",
    "Preciso de ajuda para acessar o sistema.",
    "Existe um manual para uso do sistema de ponto?"
]

# 2. Função de pré-processamento
def preprocess(text):
    stop_words = set(stopwords.words('portuguese'))
    stemmer = RSLPStemmer()

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

# 3. Aplicar pré-processamento
docs_clean = [preprocess(doc) for doc in docs]

# 4. Vetorização com TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs_clean)

# 5. Cálculo da Similaridade do Cosseno
similarity_matrix = cosine_similarity(X)

# 6. Exibir a matriz de similaridade
print("Matriz de Similaridade:")
print(similarity_matrix)

# 7. Gerar rótulos e salvar o gráfico da matriz
labels = [f"Doc{i+1}" for i in range(len(docs))]

sns.heatmap(similarity_matrix, annot=True, cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Similaridade por Cosseno")
plt.savefig("matriz_similaridade.png")
print("Gráfico salvo como 'matriz_similaridade.png'.")

# 8. Identificar os documentos mais similares e mais distintos
most_similar = np.unravel_index(np.argmax(similarity_matrix - np.eye(len(docs))), similarity_matrix.shape)
most_different = np.unravel_index(np.argmin(similarity_matrix + np.eye(len(docs))*2), similarity_matrix.shape)

print(f"\nMais similares: Documento {most_similar[0]+1} e Documento {most_similar[1]+1}")
print(f"Mais distintos: Documento {most_different[0]+1} e Documento {most_different[1]+1}")
