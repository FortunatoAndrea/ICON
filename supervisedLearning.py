from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RepeatedKFold, learning_curve, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


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


def returnBestHyperparameters(model, X, y, modelHyperparameters):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    grid_search = GridSearchCV(model, modelHyperparameters, cv=5, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    bestHyperparameters = grid_search.best_params_

    return bestHyperparameters


def trainModelKFold(model, modelName, X, y, modelBestHyperparameters):
    if (modelName == 'DecisionTree'):
        model = DecisionTreeClassifier(criterion=modelBestHyperparameters['criterion'],
                                       max_depth=modelBestHyperparameters['max_depth'],
                                       min_samples_split=modelBestHyperparameters['min_samples_split'],
                                       min_samples_leaf=modelBestHyperparameters['min_samples_leaf'])
    elif (modelName == 'RandomForest'):
        model = RandomForestClassifier(criterion=modelBestHyperparameters['criterion'],
                                       n_estimators=modelBestHyperparameters['n_estimators'],
                                       max_depth=modelBestHyperparameters['max_depth'],
                                       min_samples_split=modelBestHyperparameters['min_samples_split'],
                                       min_samples_leaf=modelBestHyperparameters['min_samples_leaf'])
    elif (modelName == 'LogisticRegression'):
        model = LogisticRegression(C=modelBestHyperparameters['C'],
                                   l1_ratio=modelBestHyperparameters['l1_ratio'],
                                   solver=modelBestHyperparameters['solver'],
                                   max_iter=modelBestHyperparameters['max_iter'])

    cv = RepeatedKFold(n_splits=5, n_repeats=5)
    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, scoring=metric, cv=cv)
        results[metric] = scores

    print("Accuracy: ", results['accuracy'])
    print("Accuracy media = ", results['accuracy'].mean())
    print("Precision Macro: ", results['precision_macro'])
    print("Precision Macro media = ", results['precision_macro'].mean())
    print("Recall Macro: ", results['recall_macro'])
    print("Recall Macro media = ", results['recall_macro'].mean())
    print("F1 Macro: ", results['f1_macro'])
    print("F1 Macro media = ", results['f1_macro'].mean())

    return model, results


def plotLearningCurve(model, modelName, x, y):
    train_sizes, train_scores, test_scores = learning_curve(model, x, y, cv=5, scoring='accuracy')
    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Stampa i valori numerici della deviazione standard e della varianza
    print(f"{modelName} - Train Error Std: {train_errors_std[-1]}, "
          f"Test Error Std: {test_errors_std[-1]}, "
          f"Train Error Var: {train_errors_var[-1]}, "
          f"Test Error Var: {test_errors_var[-1]}")

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    # Visualizza la curva di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {modelName}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()

