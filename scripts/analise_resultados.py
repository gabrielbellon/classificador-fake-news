from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(ax, y_true, y_pred, title):
    # Calcula a matriz de confusão em porcentagens
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Exibe a matriz de confusão
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Dark2', xticklabels=['0', '1'], yticklabels=['0', '1'], ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    ax.set_title(title)


def eval_models(models, X_test, y_test):
    for m in models:
        nome = m['nome']
        model = m['modelo']

        # Mostra a acurácia e AUC do modelo
        print('-' * ((80 - len(nome)) // 2), nome, '-' * ((80 - len(nome)) // 2))
        print(f'Acurácia: {accuracy_score(y_test, model.predict(X_test)):.4f}')
        print(f'AUC (area under the ROC curve): {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}')


def show_roc_curve(models, X_test, y_test):
    for m in models:
        nome = m['nome']
        model = m['modelo']

        # Adiciona os modelos no gráfico
        fpr_model, tpr_model, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc_model = auc(fpr_model, tpr_model)
        plt.plot(fpr_model, tpr_model, label=f'{nome} (AUC = {auc_model:.2f})')

    # Adiciona um classificador aleatório
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Aleatório')

    # Adiciona rótulos e legendas
    plt.xlabel('Taxa de falso positivo')
    plt.ylabel('Taxa de verdadeiro positivo')
    plt.title('ROC Curve')
    plt.legend()

    plt.show()
