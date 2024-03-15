from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def train_logistic_regression(X_train, y_train):
    param_grid = {
        'solver': ['sag', 'saga'],
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [1000]
    }

    model = LogisticRegression()

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Retorna o melhor modelo encontrado na busca de parâmetros
    return grid_search.best_estimator_


def train_bayes(X_train, y_train):
    param_grid = {
        'alpha': [0.1, 0.5, 1.0],
        'fit_prior': [True, False]
    }
    
    model = MultinomialNB()
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Retorna o melhor modelo encontrado na busca de parâmetros
    return grid_search.best_estimator_


def train_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    
    model = DecisionTreeClassifier()
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Retorna o melhor modelo encontrado na busca de parâmetros
    return grid_search.best_estimator_
