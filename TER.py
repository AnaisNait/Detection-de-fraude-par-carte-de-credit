import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


chemin= "C:\\Users\\hp\\Downloads\\M1 data scale\\S2\\TER\\Detection-de-fraude-par-carte-de-credit\\creditcard.csv"
data= pd.read_csv(chemin)
print(data.info())
print(data.head())

###Exploration des données: Analyse statistique univariée 
###Afficher des informations sur le dataframe (nb lignes, nb colonnes, nb valeurs non nulles, type)
data.info()
### Afficher toutes les colonnes du DataFrame
pd.set_option('display.max_columns', None)

print(data['Class'].value_counts())

### Afficher une analyse statistique univariée pour chaque attribut quantitatif
statistiques_univariees = data.describe()
print(statistiques_univariees)

###Prétraitement de données (valeurs aberrantes)
data.boxplot(figsize=(20,3))
plt.title('Boxplot des données')

#standardisation de données 
scaler = StandardScaler()
# Sélectionner toutes les colonnes sauf l'attribut de classe
X = data.drop(columns=['Class'])  
X_standardized = scaler.fit_transform(data)

### Nuage de points pour les variables non standardisées (data)
plt.figure(figsize=(10, 6))
for column in data.columns:
    if column != 'Class':  # Exclure la variable de classe
        plt.scatter(range(len(data)), data[column], label=column, alpha=0.5)
plt.title('Nuage de points pour les variables non standardisées')
plt.xlabel('Index de l\'échantillon')
plt.ylabel('Valeur de la variable')
plt.legend()
plt.show()

### Nuage de points pour les variables après standardisation (X_standardized)

plt.figure(figsize=(10, 6))
for i in range(X_standardized.shape[1]):  # Boucler sur toutes les colonnes de X_standardized
    plt.scatter(range(len(X_standardized)), X_standardized[:, i], label=X.columns[i], alpha=0.5)
plt.title('Nuage de points pour les variables standardisées')
plt.xlabel('Index de l\'échantillon')
plt.ylabel('Valeur de la variable standardisée')
plt.legend()
plt.show()

################################### Isolation forest ######################################
from sklearn.model_selection import train_test_split

y = data.Class 
# Diviser les données en ensembles d'apprentissage et de test pour les variables explicatives et la variable cible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import IsolationForest

# Créer et entraîner le modèle
model = IsolationForest()
#model.fit(X_train)
y_pred = model.predict(X_test) #prédiction du modèle sur les données de test
y_pred[y_pred == 1] = 0  # Prédiction correcte (normal)
y_pred[y_pred == -1] = 1  # Anomalie détectée

from sklearn.metrics import classification_report
#comparer les étiquettes prédites avec les étiquettes réelles
print(classification_report(y_test, y_pred)) 

### Suréchantillonnage 
from imblearn.over_sampling import SMOTE

# Appliquer SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Créer et entraîner le modèle Isolation Forest
model.fit(X_resampled, y_resampled)

# Prédiction du modèle sur les données de test
y_pred = model.predict(X_test)

# Transformer les prédictions pour qu'elles correspondent aux classes initiales
y_pred[y_pred == 1] = 0  # Prédiction correcte (normal)
y_pred[y_pred == -1] = 1  # Anomalie détectée

# Afficher les résultats
print(classification_report(y_test, y_pred))

###################################SVM################################# 

from sklearn.svm import SVC
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("Résultats SVM sans sur-échantillonnage :")
print(classification_report(y_test, y_pred_svm))


#Suréchantillonnage
# Créer et entraîner le modèle SVM sur les données sur-échantillonnées
svm_model.fit(X_resampled, y_resampled)
y_pred_svm_resampled = svm_model.predict(X_test)
print("Résultats SVM avec sur-échantillonnage :")
print(classification_report(y_test, y_pred_svm_resampled))

# Comparaison des résultats 
algorithms = ['Random Forest', 'Détection d\'anomalies', 'SVM']
precision_0 = [1.00, 1.00, 1.00]  # Précision pour la classe 0
recall_0 = [1.00, 0.98, 1.00]     # Rappel pour la classe 0
precision_1 = [0.93, 0.03, 0.60]  # Précision pour la classe 1
recall_1 = [0.76, 0.28, 0.30]     # Rappel pour la classe 1

# Configuration du graphique
x = np.arange(len(algorithms))  # La position des étiquettes sur l'axe des x
width = 0.2  # La largeur des barres

fig, ax = plt.subplots(figsize=(12, 6))

# Barres pour la classe 0
bars1 = ax.bar(x - width, precision_0, width, label='Précision Classe 0', color='b')
bars2 = ax.bar(x, recall_0, width, label='Rappel Classe 0', color='c')

# Barres pour la classe 1
bars3 = ax.bar(x + width, precision_1, width, label='Précision Classe 1', color='r')
bars4 = ax.bar(x + 2*width, recall_1, width, label='Rappel Classe 1', color='m')

# Ajouter des étiquettes, titre et légende
ax.set_xlabel('Algorithmes')
ax.set_ylabel('Scores')
ax.set_title('Précision et Rappel pour les Classes 0 et 1')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(algorithms)
ax.legend()

# Afficher les valeurs au-dessus des barres
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Afficher le graphique
plt.show()


