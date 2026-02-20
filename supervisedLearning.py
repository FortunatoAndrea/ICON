import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import numpy as np


#DECISION TREE
def createDecisionTree(X_train, y_train):
    # Mi fermo ad una profondità pari a 5 per rendere leggibile l'albero
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_clf.fit(X_train, y_train)

    visualizeDecisionTree(X_train, dt_clf)
    return dt_clf


def visualizeDecisionTree(X_train, dt_clf):
    # Visualizzazione grafica
    plt.figure(figsize=(20, 10))
    plot_tree(dt_clf,
              feature_names=X_train.columns,
              class_names=['0', '1', '2', '3'],
              filled=True,
              rounded=True,
              fontsize=12)
    plt.show()
    printDecisionTree(X_train, dt_clf)
    return


def printDecisionTree(X_train, dt_clf):
    tree_rules = export_text(dt_clf, feature_names=list(X_train.columns))
    print(tree_rules)
    return

#RANDOM FOREST
def createRandomForest(X_train, y_train):
    # Uso 100 Decision Tree per "votare" ìl price_range
    # Mi fermo ad una profondità di 10
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_clf.fit(X_train, y_train)

    return rf_clf


scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

def modelScores(model, X_train, y_train):
    results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=metric)
        results[metric] = scores

    return results


# Funzione che calcola Varianza, Deviazione Standard e disegna la curva di apprendimento
def visualizeLearningCurve(model_name, model, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy')

    train_errors: float = 1 - train_scores
    test_errors: float = 1 - test_scores

    train_errors_std: float = np.std(train_errors, axis=1)
    test_errors_std: float = np.std(test_errors, axis=1)
    train_errors_var: float = np.var(train_errors, axis=1)
    test_errors_var: float = np.var(test_errors, axis=1)

    print(f"{model_name} -\n "
          f"Train || Deviazione Standard: {train_errors_std[-1]}, Varianza: {train_errors_var[-1]} ,\n "
          f"Test  || Deviazione Standard: {test_errors_std[-1]}, Varianza: {test_errors_var[-1]}")

    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()

    return