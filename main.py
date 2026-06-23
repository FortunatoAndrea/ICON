import pandas as pd
from sklearn.cluster import KMeans
import unsupervisedLearning as ul
import supervisedLearning as sl
import ragionamento as r
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# Caricamento
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Creo Knowledge Base
output_file = r.creaKnowledgeBase(train)
r.scrivi_regole(output_file, train)

# Esempio: Inventiamo un telefono per ogni fascia di prezzo
nuovo_tel0 = r.inventa_telefono(0)
print(f"(Fascia 0): {nuovo_tel0}")

nuovo_tel1 = r.inventa_telefono(1)
print(f"(Fascia 1): {nuovo_tel1}")

nuovo_tel2 = r.inventa_telefono(2)
print(f"(Fascia 2): {nuovo_tel2}")

nuovo_tel3 = r.inventa_telefono(3)
print(f"(Fascia 3): {nuovo_tel3}")

###
#r.crea_nuovo_dataset()


###APPRENDIMENTO NON SUPERVSIONATO
scaler = StandardScaler()
# Prepariamo X_train (togliendo il target) e X_test (togliendo l'id)
X_train = train.drop('price_range', axis=1)
X_test = test.drop('id', axis=1)

asse_x = 'ram'
asse_y = 'battery_power'

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
price_ranges = sorted(train['price_range'].unique())

for i, pr in enumerate(price_ranges):
    # Filtriamo per fascia di prezzo
    subset = train[train['price_range'] == pr].copy()

    # Prepariamo i dati per il clustering (togliamo il target)
    features = subset.drop('price_range', axis=1)

    # Lo scaling è fondamentale: mette RAM e Batteria sulla stessa scala (0-1 o simile)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    k_test = ul.findElbow(scaled_features)

    # Applico il K-Means
    kmeans = KMeans(n_clusters=k_test, random_state=42, n_init=10)
    subset['cluster'] = kmeans.fit(scaled_features).labels_

    # Creazione del grafico usando variabili reali
    sns.scatterplot(
        x=asse_x,
        y=asse_y,
        hue='cluster',
        data=subset,
        palette='bright',
        ax=axes[i],
        s=60
    )

    axes[i].set_title(f'Fascia di Prezzo: {pr}')
    axes[i].set_xlabel(asse_x.upper())
    axes[i].set_ylabel(asse_y.upper())

plt.suptitle('Analisi Cluster per Fascia di Prezzo (Basata su RAM e Batteria)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Fondamentale per K-Means: Normalizzazione

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#REGOLA DEL GOMITO per stabilire k
k = ul.findElbow(X_train_scaled)
print("Numero di cluster (k) ideale: ", k)

# Creazione del modello
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

# Predizione sui dati di test
test_clusters = kmeans.predict(X_test_scaled)

test_pred = pd.read_csv('test.csv')
test_pred['predicted_price_range'] = test_clusters

# Cercato di predire sul training set (3=636 1=470 2=450 0=444)
new_clusters = kmeans.predict(X_train_scaled)
train_pred = X_train
train_pred['price_range'] = new_clusters

distribuzione_train = train_pred['price_range'].value_counts()
print("Distribuzione del training set (KMeans): ", distribuzione_train)

#Rappresentazione grafica della distribuzione dei dati
plt.figure(figsize=(8, 8))
plt.pie(distribuzione_train,
        labels=[f'Cluster {i}' for i in distribuzione_train.index],
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.005] * len(distribuzione_train),
        shadow=False)

plt.title('Percentuale di dati per ogni Cluster')
plt.show()



###APPRENDIMENTO SUPERVISIONATO

# Preparazione Feature e Target
X = train.drop('price_range', axis=1).to_numpy()
y = train['price_range'].to_numpy()

##########################################################################################################
#DECISION_TREE
dt_clf = DecisionTreeClassifier(random_state=42)

DecisionTreeHyperparameters = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10, 20]
}

dt_bestHyperparameters = sl.returnBestHyperparameters(dt_clf, X, y, DecisionTreeHyperparameters)

print("Migliori Iperparametri per Decision Tree: \n", dt_bestHyperparameters)

dt_clf, dt_results = sl.trainModelKFold(dt_clf, 'DecisionTree', X, y, dt_bestHyperparameters)

sl.plotLearningCurve(dt_clf, 'DecisionTree', X, y)

#dt_clf.fit(X, y)
#sl.visualizeDecisionTree(X, dt_clf)

###Decision Tree SMOTE
strategy = {0: 5000, 1: 5000, 2: 5000, 3: 5000}
smote = SMOTE(sampling_strategy=strategy, k_neighbors=5 ,random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

new_dt_clf = DecisionTreeClassifier(random_state=42)

new_dt_bestHyperparameters = sl.returnBestHyperparameters(new_dt_clf, X_resampled, y_resampled, DecisionTreeHyperparameters)
#new_dt_bestHyperparameters = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split':2, 'min_samples_leaf':2}

print("NUOVI Migliori Iperparametri per Decision Tree: \n", new_dt_bestHyperparameters)

new_dt_clf, new_dt_results = sl.trainModelKFold(new_dt_clf, 'DecisionTree', X_resampled, y_resampled, new_dt_bestHyperparameters)

sl.plotLearningCurve(new_dt_clf, 'DecisionTree', X_resampled, y_resampled)
###

##########################################################################################################
#RANDOM_FOREST
rf_clf = RandomForestClassifier(random_state=42)

RandomForestHyperparameters = {
        'criterion': ['gini', 'entropy','log_loss'],
        'n_estimators': [10, 20, 50],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10, 20]
}

rf_bestHyperparameters = sl.returnBestHyperparameters(rf_clf, X, y, RandomForestHyperparameters)

print("Migliori Iperparametri per Random Forest: \n", rf_bestHyperparameters)

rf_clf, rf_results = sl.trainModelKFold(rf_clf, 'RandomForest', X, y, rf_bestHyperparameters)

sl.plotLearningCurve(rf_clf, 'RandomForest', X, y)

###Random Forest SMOTE
strategy = {0: 5000, 1: 5000, 2: 5000, 3: 5000}
smote = SMOTE(sampling_strategy=strategy, k_neighbors=5 ,random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

new_rf_clf = RandomForestClassifier(random_state=42)

new_rf_bestHyperparameters = sl.returnBestHyperparameters(new_rf_clf, X_resampled, y_resampled, RandomForestHyperparameters)
#new_rf_bestHyperparameters = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split':10, 'min_samples_leaf':1, 'n_estimators':50}

print("NUOVI Migliori Iperparametri per Random Forest: \n", new_rf_bestHyperparameters)

new_rf_clf, new_rf_results = sl.trainModelKFold(new_rf_clf, 'RandomForest', X_resampled, y_resampled, new_rf_bestHyperparameters)

sl.plotLearningCurve(new_rf_clf, 'RandomForest', X_resampled, y_resampled)


##########################################################################################################
#LOGISTIC_REGRESSION
# La Logistic Regression richiede dati scalati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)

# Inizializza il modello
log_reg = LogisticRegression(random_state=42)

LogisticRegressionHyperparameters = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0], #lbfgs supporta solo l2 o none penalties l1_ratio=1 -> L1-penalty l1_ratio=0 -> L2-penalty
        'solver': ['lbfgs'], #liblnear non supportato per multiclass classification(n_classes >= 3)
        'max_iter': [100000,150000]}

lr_bestHyperparameters = sl.returnBestHyperparameters(log_reg, X_train_scaled, y, LogisticRegressionHyperparameters)

print("Migliori Iperparametri per Logistic Regression: \n", lr_bestHyperparameters)

log_reg, lr_results = sl.trainModelKFold(log_reg, 'LogisticRegression', X_train_scaled, y, lr_bestHyperparameters)

sl.plotLearningCurve(log_reg, 'LogisticRegression', X_train_scaled, y)


###Logistic Regression SMOTE
###
strategy = {0: 5000, 1: 5000, 2: 5000, 3: 5000}
smote = SMOTE(sampling_strategy=strategy, k_neighbors=5 ,random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y)

new_log_reg = LogisticRegression(random_state=42)

new_lr_bestHyperparameters = sl.returnBestHyperparameters(new_log_reg, X_resampled, y_resampled, LogisticRegressionHyperparameters)

print("NUOVI Migliori Iperparametri per Logistic Regression: \n", new_lr_bestHyperparameters)

new_log_reg, new_lr_results = sl.trainModelKFold(new_log_reg, 'LogisticRegression', X_resampled, y_resampled, new_lr_bestHyperparameters)

sl.plotLearningCurve(new_log_reg, 'LogisticRegression', X_resampled, y_resampled)


###GRAFICO A BARRE
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
index = np.arange(len(scoring_metrics))
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2

dt = [dt_results['accuracy'].mean(),
      dt_results['precision_macro'].mean(),
      dt_results['recall_macro'].mean(),
      dt_results['f1_macro'].mean()]
rf = [rf_results['accuracy'].mean(),
      rf_results['precision_macro'].mean(),
      rf_results['recall_macro'].mean(),
      rf_results['f1_macro'].mean()]
lr = [lr_results['accuracy'].mean(),
      lr_results['precision_macro'].mean(),
      lr_results['recall_macro'].mean(),
      lr_results['f1_macro'].mean()]

rects1 = ax.bar(index - bar_width, dt, bar_width, label='DecisionTree')
rects2 = ax.bar(index, rf, bar_width, label='RandomForest')
rects3 = ax.bar(index + bar_width, lr, bar_width, label='LogisticRegression')

ax.set_xlabel('Modelli')
ax.set_ylabel('Punteggi medi')
ax.set_xticks(index)
ax.set_xticklabels(scoring_metrics)

ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')
ax.bar_label(rects3, padding=3, fmt='%.3f')

plt.title('Punteggio medio per ogni modello')
fig.tight_layout()
plt.show()


###GRAFICO A BARRE DOPO SMOTE
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
index = np.arange(len(scoring_metrics))
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2

dt = [new_dt_results['accuracy'].mean(),
      new_dt_results['precision_macro'].mean(),
      new_dt_results['recall_macro'].mean(),
      new_dt_results['f1_macro'].mean()]
rf = [new_rf_results['accuracy'].mean(),
      new_rf_results['precision_macro'].mean(),
      new_rf_results['recall_macro'].mean(),
      new_rf_results['f1_macro'].mean()]
lr = [new_lr_results['accuracy'].mean(),
      new_lr_results['precision_macro'].mean(),
      new_lr_results['recall_macro'].mean(),
      new_lr_results['f1_macro'].mean()]

rects1 = ax.bar(index - bar_width, dt, bar_width, label='DecisionTree')
rects2 = ax.bar(index, rf, bar_width, label='RandomForest')
rects3 = ax.bar(index + bar_width, lr, bar_width, label='LogisticRegression')

ax.set_xlabel('Modelli')
ax.set_ylabel('Punteggi medi')
ax.set_xticks(index)
ax.set_xticklabels(scoring_metrics)

ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')
ax.bar_label(rects3, padding=3, fmt='%.3f')

plt.title('Punteggio medio per ogni modello')
fig.tight_layout()
plt.show()
