import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


chemin= "C:\\Users\\hp\\Downloads\\M1 data scale\\S2\\TER\\Detection-de-fraude-par-carte-de-credit\\creditcard.csv"
data= pd.read_csv(chemin)
#print(data.info())
#print(data.head())

#Exploration des données: Analyse statistique univariée 
#Afficher des informations sur le dataframe (nb lignes, nb colonnes, nb valeurs non nulles, type)
data.info()
# Afficher toutes les colonnes du DataFrame
pd.set_option('display.max_columns', None)

print(data['Class'].value_counts())

# Afficher une analyse statistique univariée pour chaque attribut quantitatif
statistiques_univariees = data.describe()
print(statistiques_univariees)

#Prétraitement de données (valeurs aberrantes)
data.boxplot(figsize=(20,3))

#standardisation de données 
scaler = StandardScaler()
# Sélectionner toutes les colonnes sauf l'attribut de classe
X = data.drop(columns=['Class'])  
X_standardized = scaler.fit_transform(data)

# Nuage de points pour les variables non standardisées (data)
plt.figure(figsize=(10, 6))
for column in data.columns:
    if column != 'Class':  # Exclure la variable de classe
        plt.scatter(range(len(data)), data[column], label=column, alpha=0.5)
plt.title('Nuage de points pour les variables non standardisées')
plt.xlabel('Index de l\'échantillon')
plt.ylabel('Valeur de la variable')
plt.legend()
plt.show()

# Nuage de points pour les variables après standardisation (X_standardized)
plt.figure(figsize=(10, 6))
for i in range(X_standardized.shape[1]):  # Boucler sur toutes les colonnes de X_standardized
    plt.scatter(range(len(X_standardized)), X_standardized[:, i], label=X.columns[i], alpha=0.5)
plt.title('Nuage de points pour les variables standardisées')
plt.xlabel('Index de l\'échantillon')
plt.ylabel('Valeur de la variable standardisée')
plt.legend()
plt.show()