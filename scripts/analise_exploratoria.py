# Arquivo com todas as funcoes e codigos referentes a analise exploratoria
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords

def show_count_by_label(data):
    label_counts = data['label'].value_counts()  # Conta quantas notícias tem por categoria

    # Exibe um gráfico de barras.
    plt.bar(label_counts.index, label_counts.values, color=['skyblue', 'salmon'])
    plt.xticks(label_counts.index, ['Notícias confiáveis', 'Notícias não confiáveis'])
    plt.ylabel('Quantidade')
    plt.title('Quantidade de notícias por categoria')
    
    # Adiciona rótulos numéricos acima de cada barra
    for i, count in enumerate(label_counts.values):
        plt.text(label_counts.index[i], count + 10, str(count), ha='center', va='bottom')
        
    plt.show()


def show_temporal_by_label(data, threshold, title):
    # Distribuição temporal dos dados de treino de acordo com o label
    date = data
    data['date'] = pd.to_datetime(data['date'])
    date['year'] = data['date'].dt.year

    # Contagem de notícias por ano e label
    count_data = date.groupby(['year', 'label']).size().reset_index(name='count')
    count_data = count_data[count_data['count'] > threshold]  # Remove notícias que possivelmente foram classificadas com data errada

    # Plotagem do gráfico de barras empilhadas
    sns.barplot(x='year', y='count', hue='label', data=count_data, palette='viridis')
    plt.xlabel('Ano')
    plt.ylabel('Número de Notícias')
    plt.title(title)
    plt.show()


def show_temporal(data, threshold, title):
    # Distribuição temporal dos dados a serem avaliados
    date = data
    data['date'] = pd.to_datetime(data['date'])
    date['year'] = data['date'].dt.year

    # Contagem de notícias por ano
    count_data = date.groupby(['year']).size().reset_index(name='count')
    count_data = count_data[count_data['count'] > threshold]  # Remove notícias que possivelmente foram classificadas com data errada

    # Plotagem do gráfico de barras empilhadas
    sns.barplot(x='year', y='count', data=count_data, palette='viridis')
    plt.xlabel('Ano')
    plt.ylabel('Número de Notícias')
    plt.title(title)
    plt.show()


def show_null(data):
    # Contagem de valores nulos nas colunas 'title' e 'content'
    null_title_counts = data['title'].isnull().sum()
    null_content_counts = data['content'].isnull().sum()

    # Criando DataFrame para as contagens de valores nulos
    null_counts_df = pd.DataFrame({'Coluna': ['Title', 'Content'], 'Quantidade Nula': [null_title_counts, null_content_counts]})

    # Exibe o gráfico de barras para valores nulos
    null_counts_df.plot(kind='bar', x='Coluna', y='Quantidade Nula', color='skyblue')
    plt.title('Quantidade de Valores Nulos por Coluna')
    plt.xlabel('Coluna')
    plt.ylabel('Quantidade')
    plt.tick_params(axis='x', rotation=0)
    plt.show()


def generate_wordcloud(text, title):
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    wordcloud = WordCloud(stopwords=stop_words, width=800, height=400, background_color='white').generate(text)
    
    # Exibe a núvem de palavas
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()
