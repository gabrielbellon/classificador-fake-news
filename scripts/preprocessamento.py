import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocessing(text):
    if not isinstance(text, str):  # Verifica se o valor é uma string
        return ""

    # normaliza os numeros 
    regex = re.compile('[0-9]+')
    text = re.sub(regex, "number", text)

    # Lower case
    text = text.lower()
    
    # remove tags HTML
    regex = re.compile('<[^<>]+>')
    text = re.sub(regex, " ", text) 
    
    # normaliza as URLs
    regex = re.compile('(http|https)://[^\s]*')
    text = re.sub(regex, "enderecoweb", text)

    # normaliza emails
    regex = re.compile('[^\s]+@[^\s]+')
    text = re.sub(regex, "enderecoemail", text)
    
    #normaliza o símbolo de dólar
    regex = re.compile('[$]+')
    text = re.sub(regex, "dolar", text)
    
    # converte todos os caracteres não-alfanuméricos em espaço
    regex = re.compile('[^A-Za-z]') 
    text = re.sub(regex, " ", text)
    
    # Remove stopwords do texto
    tokenized_text = word_tokenize(text)
    palavras_sem_stop_words = [palavra for palavra in tokenized_text if palavra not in stop_words]
    text = ' '.join(palavras_sem_stop_words)

    # substitui varios espaçamentos seguidos em um só
    text = ' '.join(text.split())

    return text
