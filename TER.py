import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


chemin= "C:\\Users\\hp\\Downloads\\M1 data scale\\S2\\TER\\Detection-de-fraude-par-carte-de-credit\\creditcard.csv"
data= pd.read_csv(chemin)
print(data.info())
print(data.head())

#Exploration des données: Analyse statistique univariée 
#Afficher des informations sur le dataframe (nb lignes, nb colonnes, nb valeurs non nulles, type)
data.info()
#Afficher toutes les colonnes du DataFrame
pd.set_option('display.max_columns', None)

print(data['Class'].value_counts())

#Afficher une analyse statistique univariée pour chaque attribut quantitatif
statistiques_univariees = data.describe()
print(statistiques_univariees)

#Prétraitement de données (valeurs aberrantes)
data.boxplot(figsize=(20,3))
plt.title('Boxplot des données')

#standardisation de données 
scaler = StandardScaler()
#Sélectionner toutes les colonnes sauf l'attribut de classe
X = data.drop(columns=['Class'])  
X_standardized = scaler.fit_transform(data)

#Nuage de points pour les variables non standardisées (data)
plt.figure(figsize=(10, 6))
for column in data.columns:
    if column != 'Class':  # Exclure la variable de classe
        plt.scatter(range(len(data)), data[column], label=column, alpha=0.5)
plt.title('Nuage de points pour les variables non standardisées')
plt.xlabel('Index de l\'échantillon')
plt.ylabel('Valeur de la variable')
plt.legend()
plt.show()

#Nuage de points pour les variables après standardisation (X_standardized)

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

#Créer et entraîner le modèle
model = IsolationForest()
model.fit(X_train)
y_pred = model.predict(X_test) #prédiction du modèle sur les données de test
y_pred[y_pred == 1] = 0  # Prédiction correcte (normal)
y_pred[y_pred == -1] = 1  # Anomalie détectée

from sklearn.metrics import classification_report
#comparer les étiquettes prédites avec les étiquettes réelles
print(classification_report(y_test, y_pred)) 

#Suréchantillonnage 
from imblearn.over_sampling import SMOTE

#Appliquer SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

#Créer et entraîner le modèle Isolation Forest#model.fit(X_resampled, y_resampled)

#Prédiction du modèle sur les données de test
y_pred = model.predict(X_test)

#Transformer les prédictions pour qu'elles correspondent aux classes initiales
y_pred[y_pred == 1] = 0  # Prédiction correcte (normal)
y_pred[y_pred == -1] = 1  # Anomalie détectée

#Afficher les résultats
print(classification_report(y_test, y_pred))

###################################SVM################################# 

from sklearn.svm import SVC
#Créer le modèle SVM
svm_model = SVC(kernel='linear')
#Entraîner le modèle sur les données d'entraînement
svm_model.fit(X_train, y_train)
#Faire des prédictions sur les données de test
y_pred_svm = svm_model.predict(X_test)
#Afficher les paramètres d'évaluation pour le SVM
print(classification_report(y_test, y_pred_svm))

svm_model.fit(X_resampled, y_resampled)
y_pred_svm = svm_model.predict(X_test)
print(classification_report(y_test, y_pred_svm))


# Obtenir les scores de décision du SVM pour chaque transaction
decision_scores = svm_model.decision_function(X_test)

# Créer un DataFrame avec les scores de décision et les classes réelles
df_scores = pd.DataFrame(decision_scores, columns=['Decision_Score'])
df_scores['Class'] = y_test.values

# Créer le boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Decision_Score', data=df_scores)
plt.title('Boxplot des scores de décision du SVM pour les transactions frauduleuses et non frauduleuses')
plt.xlabel('Classe')
plt.ylabel('Score de décision')
plt.show()