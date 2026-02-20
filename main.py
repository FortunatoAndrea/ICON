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
from sklearn.metrics import accuracy_score

# Caricamento
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Creo Knowledge Base
output_file = r.creaKnowledgeBase(train)

r.scrivi_regole(output_file, train)

###
#Ignoro le colonne booleane
colonne = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep',
           'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width',
           'ram', 'sc_h', 'sc_w', 'talk_time']


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
r.crea_nuovo_dataset()


###APPRENDIMENTO NON SUPERVSIONATO
scaler = StandardScaler()
# Prepariamo X_train (togliendo il target) e X_test (togliendo l'id)
X_train = train.drop('price_range', axis=1)
X_test = test.drop('id', axis=1)

import seaborn as sns

# Scegliamo due variabili "umane" per gli assi del grafico
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
#print(test_pred['predicted_price_range'])
#test_pred[['id', 'predicted_price_range']].to_csv('predictions.csv', index=False)

# Distribuzione del training set (0=500,1=500,2=500,3=500)
print("Distribuzione del training set: ")
print(train['price_range'].value_counts())
# Distribuzione della predizione sul test set (0=200,1=235,2=239,3=326)
distribuzione_test = test_pred['predicted_price_range'].value_counts()
print("Distribuzione del test set (KMeans): ")
print(distribuzione_test)

# Rappresentazione grafica della distribuzione dei dati
plt.figure(figsize=(8, 8))
plt.pie(distribuzione_test,
        labels=[f'Cluster {i}' for i in distribuzione_test.index],
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.005] * len(distribuzione_test),
        shadow=False)

plt.title('Percentuale di dati per ogni Cluster (test set)')
plt.show()

# Cercato di predire sul training set (3=636 1=470 2=450 0=444)
new_clusters = kmeans.predict(X_train_scaled)
train_pred = X_train
train_pred['price_range'] = new_clusters
#train_pred[['price_range']].to_csv('train_predictions.csv', index=False)
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

X_train = train.drop('price_range', axis=1)
y_train = train['price_range']
X_test = test.drop('id', axis=1)


##########################################################################################################
#DECISION_TREE
from sklearn.model_selection import cross_val_score

dt_clf = sl.createDecisionTree(X_train, y_train)

y_pred = dt_clf.predict(X_test)

# Creo una struttura dati contenente le predizioni del Decision Tree sul dataset di Test
predictions_dt = pd.DataFrame({
    'id': test.id.values,
    'price_range': y_pred,
})

# Salvo su file le predizioni
predictions_dt.to_csv('Decision_Tree_predictions.csv', index=False)
print("File delle previsioni del Decision Tree creato con successo!")


#Dato che non posso verificare l'accuratezza delle predizioni sul dataset di Test,
#eseguo il k-fold sul dataset di Train
scores = cross_val_score(dt_clf, X_train, y_train)
print(f"Accuratezza media stimata Decision Tree: {scores.mean() * 100:.2f}%")


##########################################################################################################
#RANDOM_FOREST

# Creazione e Addestramento del Modello
rf_clf = sl.createRandomForest(X_train, y_train)

# Predizione
# Ora il modello sa esattamente che 0, 1, 2, 3 sono le classi reali
price_range_preds = rf_clf.predict(X_test)

# Creo una struttura dati contenente le predizioni della Random Forest sul dataset di Test
predictions_rf = pd.DataFrame({
    'id': test['id'],
    'price_range': price_range_preds
})

# Salvo su file le predizioni
predictions_rf.to_csv('Random_Forest_predictions.csv', index=False)
print("File delle previsioni della Random Forest creato con successo!")

#VALIDAZIONE RANDOM_FOREST
from sklearn.model_selection import train_test_split
# Split del Training Set per la validazione
X = train.drop('price_range', axis=1)
y = train['price_range']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento del Classificatore
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Verifica Accuratezza
val_predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)

print(f"Accuratezza media stimata Random Forest: {accuracy * 100:.2f}%")

##########################################################################################################
#LOGISTIC_REGRESSION

# Split per Validazione
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# La Logistic Regression richiede dati scalati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Inizializza il modello
log_reg = LogisticRegression(random_state=42, max_iter=1000)

# Addestramento
log_reg.fit(X_train_scaled, y_train)

# Predizione per la Validazione
y_pred_log = log_reg.predict(X_val_scaled)

# Validazione
print(f"Accuratezza media stimata Logistic Regression: {accuracy_score(y_val, y_pred_log) * 100:.2f}%")

# Creo una struttura dati contenente le feature con associato i loro coefficienti
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_reg.coef_[3]
}).sort_values(by='Coefficient', ascending=False)

print("Coefficienti: \n",coef_df)

lr_pred = log_reg.predict(X_test_scaled)

# Creo una struttura dati contenente le predizioni della Logistic Regression sul dataset di Test
predictions_lr = pd.DataFrame({
    'id': test['id'],
    'price_range': lr_pred
})
predictions_lr.to_csv('Logistic_Regression_predictions.csv', index=False)
print("File delle previsioni della Logistic Regression creato con successo!")

models = {
    'DecisionTree':{
        'accuracy_list':[],
        'precision_list':[],
        'recall_list':[],
        'f1_list':[]
    },
    'RandomForest':{
        'accuracy_list':[],
        'precision_list':[],
        'recall_list':[],
        'f1_list':[]
    },
    'LogisticRegression':{
        'accuracy_list':[],
        'precision_list':[],
        'recall_list':[],
        'f1_list':[]
    }
}

scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

##########################################################################################################
# VALUTAZIONE DELLE PRESTAZIONI
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

cv = RepeatedKFold(n_splits=5, n_repeats=5)

m = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
}

all_scores = {}

for name, model in m.items():
    if model == 'LogisticRegression':
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv)
    else:
        scores = cross_val_score(model, X_train, y_train, cv=cv)
    all_scores[name] = scores


df_results = pd.DataFrame(all_scores)
df_results.index = [[f'Rep {i//5 + 1} - Fold {i%5 + 1}' for i in range(25)]]

print(df_results)
print("\n--- STATISTICHE FINALI ---")
summary = df_results.agg(['mean', 'std', 'min', 'max']).T
print(summary)



###
dt_scores = sl.modelScores(dt_clf, X_train, y_train)
rf_scores = sl.modelScores(rf_clf, X_train, y_train)
lr_scores = sl.modelScores(log_reg, X_train_scaled, y_train)

models['DecisionTree']['accuracy_list'] = dt_scores['accuracy']
models['DecisionTree']['precision_list'] = dt_scores['precision_macro']
models['DecisionTree']['recall_list'] = dt_scores['recall_macro']
models['DecisionTree']['f1_list'] = dt_scores['f1_macro']

models['RandomForest']['accuracy_list'] = rf_scores['accuracy']
models['RandomForest']['precision_list'] = rf_scores['precision_macro']
models['RandomForest']['recall_list'] = rf_scores['recall_macro']
models['RandomForest']['f1_list'] = rf_scores['f1_macro']

models['LogisticRegression']['accuracy_list'] = lr_scores['accuracy']
models['LogisticRegression']['precision_list'] = lr_scores['precision_macro']
models['LogisticRegression']['recall_list'] = lr_scores['recall_macro']
models['LogisticRegression']['f1_list'] = lr_scores['f1_macro']

mean_scores = {
    'DecisionTree':{
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float
    },
    'RandomForest':{
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float
    },
    'LogisticRegression':{
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float
    }
}

for name in models:
    accuracy_score = models[name]['accuracy_list'].mean()
    #print(f"{name} Accuracy: {accuracy_score:.3f}")
    mean_scores[name]['accuracy'] = accuracy_score

    precision_score = models[name]['precision_list'].mean()
    #print(f"{name} Precision: {precision_score:.3f}")
    mean_scores[name]['precision'] = precision_score

    recall_score = models[name]['recall_list'].mean()
    #print(f"{name} Recall: {recall_score:.3f}")
    mean_scores[name]['recall'] = recall_score

    f1_score = models[name]['f1_list'].mean()
    #print(f"{name} F1: {f1_score:.3f}")
    mean_scores[name]['f1'] = f1_score

# Visualizzazione grafica degli score
dt = [mean_scores['DecisionTree']['accuracy'],
             mean_scores['DecisionTree']['precision'],
             mean_scores['DecisionTree']['accuracy'],
             mean_scores['DecisionTree']['f1']]
rf = [mean_scores['RandomForest']['accuracy'],
             mean_scores['RandomForest']['precision'],
             mean_scores['RandomForest']['accuracy'],
             mean_scores['RandomForest']['f1']]
lr = [mean_scores['LogisticRegression']['accuracy'],
             mean_scores['LogisticRegression']['precision'],
             mean_scores['LogisticRegression']['accuracy'],
             mean_scores['LogisticRegression']['f1']]

index = np.arange(len(scoring_metrics))
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2

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

###
#Curva di apprendimento (DECISION TREE)
sl.visualizeLearningCurve('DecisionTree', dt_clf, X_train, y_train)
sl.visualizeLearningCurve('RandomForest', rf_clf, X_train, y_train)
sl.visualizeLearningCurve('Logistic Regression', log_reg, X_train_scaled, y_train)