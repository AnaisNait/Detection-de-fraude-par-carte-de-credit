# Détection de fraude par carte de crédit
Ce projet vise à analyser et visualiser un dataset de transactions par carte de crédit afin d'identifier les transactions frauduleuses parmi celles-ci et ceci en utilisant un algorithme de machine learning. 

# 1- Introduction  
Le secteur financier est confronté à des défis liés à la sécurité des transactions, en particulier dans le contexte de la montée en puissance de la numérisation des services bancaires. La fraude par carte de crédit demeure une menace persistante, nécessitant des solutions innovantes pour préserver l'intégrité des transactions et la confiance des utilisateurs. Dans cette optique, ce projet traite un ensemble de données de transactions par carte de crédit afin de détecter les transactions frauduleuses parmi celles-ci, en exploitant les capacités de l'apprentissage automatique.  
### Contexte 
La fraude bancaire a connu une évolution remarquable au fil des décennies. Des méthodes traditionnelles, telles que les contrefaçons de chèques et les vols de cartes, ont cédé la place à des pratiques plus sophistiquées dans le monde numérique. L'émergence de la cybercriminalité a donné naissance à des formes de fraude complexes, exploitant les failles des systèmes pour dérober des informations sensibles.  
La fraude par carte de crédit demeure l'une des manifestations les plus répandues et pernicieuses de la fraude financière. Des attaques telles que le clonage de cartes, les achats non autorisés et le vol d'informations personnelles persistent, malgré les mesures de sécurité en place. L'utilisation croissante des services bancaires en ligne et des transactions électroniques expose davantage les utilisateurs à ces risques.  

# 2- État de l'Art de la Fraude Bancaire  
La détection de la fraude bancaire a connu une transformation significative au fil des décennies, s'adaptant aux évolutions technologiques et aux changements dans les modèles de comportement des fraudeurs. Cette section offre une rétrospective sur les différentes approches qui ont marqué l'histoire de la détection de fraude.  

### Antiquité à Moyen Âge :  
Les premières formes de fraudes financières étaient souvent liées à la contrefaçon de pièces de monnaie et à la falsification de documents.  
### 19ème siècle :   
L'introduction des chèques a donné lieu à des fraudes telles que la falsification de signatures.  
Les banques ont commencé à embaucher des experts en écriture pour vérifier l'authenticité des signatures.
Cependant, cette approche était limitée par la vitesse de traitement et la subjectivité des évaluations.  
### Début du 20ème siècle :  
L'avènement des transactions électroniques a introduit de nouveaux défis. Les premières formes de fraude par cartes de crédit ont émergé.  
Les banques ont commencé à utiliser des méthodes statistiques pour détecter des modèles inhabituels dans les transactions.  
### Années 1960-1970 :  
Les banques ont commencé à utiliser des systèmes de détection automatisés pour surveiller les activités financières suspectes. Cela a souvent été basé sur des règles prédéfinies et des modèles statistiques.  
Les modèles basés sur la régression logistique étaient parmi les approches les plus utilisées pour identifier les transactions potentiellement frauduleuses. Ces modèles pouvaient prendre en compte divers paramètres, tels que le montant, l'emplacement, l'heure, etc., et attribuer des scores de probabilité à ces transactions. Cependant, ils étaient limités dans leur capacité à capturer des schémas de fraude complexes, en particulier lorsque les relations entre les variables sont non linéaires.   
### Années 1990 à aujourd'hui :    
L'avènement d'Internet a conduit à des fraudes en ligne, telles que le vol d'identité et la fraude par phishing.
Les banques ont développé des outils de détection avancés basés sur l'intelligence artificielle et l'apprentissage automatique pour détecter les schémas frauduleux.  

#### _Méthodes d'apprentissage automatique:_  
* Machines à Vecteurs de Support (SVM):  
Elles permettent de créer des frontières de décision non linéaires. Elles sont efficaces pour traiter des ensembles de données où les classes ne sont pas linéairement séparables.
Les SVM peuvent être utilisées pour séparer les transactions frauduleuses des transactions légitimes en construisant des hyperplans dans un espace multidimensionnel séparant les transactions en deux classes.  
* Random Forest :  
Fonctionne en construisant plusieurs arbres de décision lors de l'entraînement et en les combinant pour obtenir une prédiction plus robuste.
* Clustering:   
Les méthodes de clustering sont utilisées pour identifier des groupes de transactions similaires et détecter des anomalies parmi ces groupes.  
#### _Méthodes d'intelligence artificielle:_   
* Réseaux de Neurones Artificiels:  
Les réseaux de neurones peuvent apprendre des représentations complexes à partir de données transactionnelles, permettant la détection de schémas de fraude non linéaires.  

# 3- Etude du dataset à utiliser:  
L’ensemble de données contient les transactions effectuées par carte de crédit en septembre 2013 par les titulaires de cartes européens.  
Cet ensemble de données présente les transactions qui ont eu lieu en deux jours, où nous avons 492 fraudes sur 284807 transactions. L’ensemble de données est très déséquilibré, la classe positive (fraudes) représente 0,172 % de toutes les transactions.  
Le dataset contient des variables d'entrée numériques qui sont le résultat d'une transformation ACP (Les variables de V1 à V28). Malheureusement, pour des raisons de confidentialité, les caractéristiques originales et plus d'informations sur le contexte des données ne sont pas fournies.  
Les seules variables qui n'ont pas été transformées avec l'ACP sont "Time" et "Amount". La variable "Time" considère le temps écoulé en secondes entre chaque transaction et la première transaction de l'ensemble de données. La variable "Amount" est le montant de la transaction. Enfin une variable "Class" est utilisée et permet de sélectionner les transactions frauduleuses et les transactions non frauduleuses, telle qu'elle prend la valeur 1 en cas de fraude et 0 dans le cas contraire.  

# 4- Etude des travaux réalisés sur le dataset:  
Dans cette section, nous allons explorer  les différents algorithmes de machine learning qui ont été utilisés dans des études antérieures du même dataset. Chaque algorithme offre des approches uniques pour modéliser et extraire des informations à partir des données  
### Random Forest  
Random Forest est un algorithme d'apprentissage supervisé qui utilise un ensemble d'arbres de décision pour effectuer des prédictions. Il est particulièrement adapté aux problèmes de classification et de régression.   
* Explication du choix du modèle :
Il a été utilisé dans le but d'identifier quelles sont les features qui ont été déterminantes pour l’obtention d’une prédiction, offrant ainsi une meilleure visibilité sur l'impact des variables sur les prédictions.  
* Résultats obtenus : 
Dans le cas de notre dataset,  Random Forest a réussi à généraliser efficacement les motifs présents dans les données sans surajustement excessif, ce qui a conduit à des prédictions précises. Le modèle possède une précision de 1, ce qui est un trés bon résultat.  
Malgré le déséquilibre observé dans notre dataset (une répartition inégale des classes où certaines classes sont moins représentées que d'autres), les résultats obtenus avec Random Forest sont jugés satisfaisants.  
En revanche, les tentatives d'amélioration des performances des algorithmes par des méthodes de sur-échantillonnage ou de sous-échantillonnage n'ont pas conduit à une amélioration significative des résultats. Ces méthodes ont été utilisées pour équilibrer la distribution des classes dans le dataset afin de mieux traiter le déséquilibre de classe, mais dans notre cas, elles n'ont pas apporté de bénéfices supplémentaires en termes de performance des algorithmes.  
En conclusion, Random Forest s'est révélé être un choix efficace pour résoudre la détection des transactions frauduleuses des transactions non frauduleuses, malgré le déséquilibre dans le dataset. Les résultats obtenus avec cet algorithme ont été satisfaisants, démontrant ainsi la robustesse de Random Forest.  

### Arbres de décision  
* Explication du choix du modèle :  
Un arbre de décision est une méthode de classification qui permettra dans notre cas de déterminer pour chaque variable un seuil a partir duquel il est plus probable d'obtenir une transaction frauduleuse et ce de manière ordonnée et hiérarchisée.  
* Résultats obtenus :    
Le modèle a un taux d'erreur  proche de 0 et des pourcentages elevés de specificité et sensitivité, proches de 1, que ce soit pour le fichier d'entrainement ou le fichier test. il a été conclu que 95.06 % des valeurs Class sont justement prédite.  

### Régression linéaire  
* Explication du choix du modèle :  
Dans ce cas, 2 méthodes de régression logistique ont été utilisées afin de découvrir la plus adaptée et la plus efficace parmi les deux. La Weighted Logistic Regression est une variante de la régression logistique qui attribue des poids différents aux exemples de données en fonction de leur déséquilibre, donnant ainsi un poids à la classe qui possède le moins de données afin de réduire l'impact du déséquilibre.   

* Résultats obtenus :   
En utilisant La Weighted Logistic Regression, un excellent résultat avec 100% de réussite pour chaque classe a été obtenu, mieux qu'en appliquant la régression logiqtique classique.    
Dans le cas de données déséquilibrées (le cas de notre jeu de données), il a été déduit que la Weighted Logistic Regression fonctionne mieux que la régression logistique classique.  

# 5- Exploration des données:  
Nous allons afficher des informations sur le DataFrame (le nombre total d'entrées, le nombre de colonnes, le nom des colonnes, le nombre de valeurs non nulles par colonne et le type de données de chaque colonne)  
```
data.info()  
```
    RangeIndex: 284807 entries, 0 to 284806  
    Data columns (total 31 columns):  
         Column  Non-Null Count  Dtype   
     0    Time    284807 non-null  float64   
     1    V1      284807 non-null  float64   
     2    V2      284807 non-null  float64  
     3    V3      284807 non-null  float64  
     4    V4      284807 non-null  float64  
     5    V5      284807 non-null  float64  
     6    V6      284807 non-null  float64  
     7    V7      284807 non-null  float64  
     8    V8      284807 non-null  float64  
     9    V9      284807 non-null  float64  
     10   V10     284807 non-null  float64   
     11   V11     284807 non-null  float64  
     12   V12     284807 non-null  float64  
     13   V13     284807 non-null  float64  
     14   V14     284807 non-null  float64  
     15   V15     284807 non-null  float64  
     16   V16     284807 non-null  float64  
     17   V17     284807 non-null  float64   
     18   V18     284807 non-null  float64   
     19   V19     284807 non-null  float64   
     20   V20     284807 non-null  float64   
     21   V21     284807 non-null  float64  
     22   V22     284807 non-null  float64  
     23   V23     284807 non-null  float64  
     24   V24     284807 non-null  float64  
     25   V25     284807 non-null  float64  
     26   V26     284807 non-null  float64  
     27   V27     284807 non-null  float64  
     28   V28     284807 non-null  float64  
     29   Amount  284807 non-null  float64  
     30   Class   284807 non-null  int64    
     dtypes: float64(30), int64(1)  
     memory usage: 67.4 MB  

> À partir des informations fournies par data.info(), nous pouvons observer les caractéristiques suivantes de l'ensemble de données :  
> - Il y a un total de 284 807 entrées (lignes) dans le DataFrame.  
> - Il y a 31 colonnes au total, chaque colonne représentant une variable différente.  
> - Les colonnes vont de 0 à 30 et sont étiquetées avec leur nom (Time, V1, V2, ..., V28, Amount, Class).  
> - Toutes les colonnes ont 284 807 valeurs non nulles, ce qui signifie qu'il n'y a pas de valeurs manquantes dans l'ensemble de données.   
> - Les types de données des colonnes sont principalement des nombres à virgule flottante (float64) pour les variables continues, et une colonne est de type entier (int64), qui est la colonne de l'attribut (Class).  

Nous allons examiner les fréquences des différentes valeurs (0,1) de l'attribut Class afin de détecter s'il y a un déséquilibre ou non  

```
print(data['Class'].value_counts())  
```

    Class  
    0       284315  
    1       492  
    Name: count, dtype: int64  

> On remarque que la valeur 0 est beaucoup plus fréquente que 1, cela indique un déséquilibre dans les valeurs de l'attribut Class
### Analyse statistique univariée    
Dans cette partie, afin de comprendre les caractéristiques de chaque variable et leur distribution, nous allons calculer les statistiques descriptives pour chaque colonne du dataframe.  
```
statistiques_univariees = data.describe()  
```
 
Ces statistiques incluent la moyenne, l'écart-type, le minimum, le maximum, le premier quartile (25%), le deuxième quartile (médiane), et le troisième quartile (75%) pour chaque donnée.  

              Time            V1            V2               V3               V4  \                              
    count  284807.000000  2.848070e+05   2.848070e+05    2.848070e+05   2.848070e+05  
    mean   94813.859575   1.168375e-15   3.416908e-16   -1.379537e-15   2.074095e-15   
    std    47488.145955   1.958696e+00   1.651309e+00    1.516255e+00   1.415869e+00     
    min    0.000000       -5.640751e+01  -7.271573e+01  -4.832559e+01  -5.683171e+00     
    25%    54201.500000   -9.203734e-01  -5.985499e-01  -8.903648e-01  -8.486401e-01     
    50%    84692.000000   1.810880e-02   6.548556e-02    1.798463e-01  -1.984653e-02     
    75%    139320.500000  1.315642e+00   8.037239e-01    1.027196e+00   7.433413e-01     
    max    172792.000000  2.454930e+00   2.205773e+01    9.382558e+00   1.687534e+01      

                V5            V6            V7            V8            V9  \    
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05     
    mean   9.604066e-16  1.487313e-15 -5.556467e-16  1.213481e-16 -2.406331e-15     
    std    1.380247e+00  1.332271e+00  1.237094e+00  1.194353e+00  1.098632e+00     
    min   -1.137433e+02 -2.616051e+01 -4.355724e+01 -7.321672e+01 -1.343407e+01     
    25%   -6.915971e-01 -7.682956e-01 -5.540759e-01 -2.086297e-01 -6.430976e-01     
    50%   -5.433583e-02 -2.741871e-01  4.010308e-02  2.235804e-02 -5.142873e-02     
    75%    6.119264e-01  3.985649e-01  5.704361e-01  3.273459e-01  5.971390e-01     
    max    3.480167e+01  7.330163e+01  1.205895e+02  2.000721e+01  1.559499e+01     

                V10           V11           V12           V13           V14  \  
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05     
    mean   2.239053e-15  1.673327e-15 -1.247012e-15  8.190001e-16  1.207294e-15     
    std    1.088850e+00  1.020713e+00  9.992014e-01  9.952742e-01  9.585956e-01     
    min   -2.458826e+01 -4.797473e+00 -1.868371e+01 -5.791881e+00 -1.921433e+01     
    25%   -5.354257e-01 -7.624942e-01 -4.055715e-01 -6.485393e-01 -4.255740e-01     
    50%   -9.291738e-02 -3.275735e-02  1.400326e-01 -1.356806e-02  5.060132e-02     
    75%    4.539234e-01  7.395934e-01  6.182380e-01  6.625050e-01  4.931498e-01     
    max    2.374514e+01  1.201891e+01  7.848392e+00  7.126883e+00  1.052677e+01     

               V15           V16           V17           V18           V19  \  
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05     
    mean   4.887456e-15  1.437716e-15 -3.772171e-16  9.564149e-16  1.039917e-15     
    std    9.153160e-01  8.762529e-01  8.493371e-01  8.381762e-01  8.140405e-01     
    min   -4.498945e+00 -1.412985e+01 -2.516280e+01 -9.498746e+00 -7.213527e+00     
    25%   -5.828843e-01 -4.680368e-01 -4.837483e-01 -4.988498e-01 -4.562989e-01     
    50%    4.807155e-02  6.641332e-02 -6.567575e-02 -3.636312e-03  3.734823e-03     
    75%    6.488208e-01  5.232963e-01  3.996750e-01  5.008067e-01  4.589494e-01     
    max    8.877742e+00  1.731511e+01  9.253526e+00  5.041069e+00  5.591971e+00     

            V20           V21           V22           V23           V24  \  
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05     
    mean   6.406204e-16  1.654067e-16 -3.568593e-16  2.578648e-16  4.473266e-15     
    std    7.709250e-01  7.345240e-01  7.257016e-01  6.244603e-01  6.056471e-01     
    min   -5.449772e+01 -3.483038e+01 -1.093314e+01 -4.480774e+01 -2.836627e+00     
    25%   -2.117214e-01 -2.283949e-01 -5.423504e-01 -1.618463e-01 -3.545861e-01     
    50%   -6.248109e-02 -2.945017e-02  6.781943e-03 -1.119293e-02  4.097606e-02     
    75%    1.330408e-01  1.863772e-01  5.285536e-01  1.476421e-01  4.395266e-01     
    max    3.942090e+01  2.720284e+01  1.050309e+01  2.252841e+01  4.584549e+00     

            V25           V26           V27           V28               Amount  \  
    count  2.848070e+05   2.848070e+05   2.848070e+05     2.848070e+05      284807.000000     
    mean   5.340915e-16   1.683437e-15   -3.660091e-16    -1.227390e-16     88.349619     
    std    5.212781e-01   4.822270e-01   4.036325e-01     3.300833e-01      250.120109     
    min   -1.029540e+01  -2.604551e+00   -2.256568e+01    -1.543008e+01     0.000000     
    25%   -3.171451e-01  -3.269839e-01   -7.083953e-02    -5.295979e-02     5.600000     
    50%    1.659350e-02  -5.213911e-02   1.342146e-03    1.124383e-02     22.000000     
    75%    3.507156e-01   2.409522e-01   9.104512e-02    7.827995e-02     77.165000     
    max    7.519589e+00   3.517346e+00   3.161220e+01      3.384781e+01     25691.160000     

            Class    
    count       284807.000000    
    mean        0.001727    
    std         0.041527    
    min         0.000000    
    25%         0.000000    
    50%         0.000000    
    75%         0.000000    
    max         1.000000  

> Les données semblent être centrées autour de zéro pour de nombreuses variables, ce qui suggère qu'elles pourraient être normalisées ou standardisées. 
> Les écarts-types indiquent des niveaux de dispersion différents pour chaque variable, certaines ayant une dispersion plus grande que d'autres.  
 
### Prétraitement des données  
Le code ci-dessous, permet de dessiner un boxplot pour identifier les colonnes du Dataset qui contiennent des valeurs aberrantes.    
```
data.boxplot(figsize=(20,3))
```
[![boxplot](https://github.com/AnaisNait/Detection-de-fraude-par-carte-de-credit/assets/103700341/4a907d2c-9103-46f1-b9c7-521ea19ab42f)](#)

> Il n'y a pas de points situés en dehors de la boite à moustaches. Tous les points sont à l'intérieur, donc il n'y a pas de valeurs aberrantes.    

### Standardisation 
La standardisation vise à transformer les données de telle sorte qu'elles aient une moyenne de 0 et un écart-type de 1.  
Comme nous voulons utiliser l'algo de détection d'anomalies, nous standardisons uniquement les les variables d'entrée afin de rendre les calculs plus efficaces et d'améliorer la performance de m'algorithme utilisé. La variable de classe, qui indique si une observation est une anomalie ou non, n'a pas besoin d'être standardisée.  
```
scaler = StandardScaler()
X = data.drop(columns=['Class'])  
X_standardized = scaler.fit_transform(data)
```
Nous dessinons ci-dessous le nuage de points des variables avant la standardisation  
``` 
plt.figure(figsize=(10, 6))
for column in data.columns:
    if column != 'Class':  # Exclure la variable de classe
        plt.scatter(range(len(data)), data[column], label=column, alpha=0.5)
plt.title('Nuage de points pour les variables non standardisées')
plt.xlabel('Index de l\'échantillon')
plt.ylabel('Valeur de la variable')
plt.legend()
plt.show()
```
[![NP_avant_standardisation](https://github.com/AnaisNait/Detection-de-fraude-par-carte-de-credit/assets/103700341/043fc586-3a59-4378-9332-3eddf9f9906a)](#)

Nous dessinons ci-dessous le nuage de points des variables après la standardisation  
```
plt.figure(figsize=(10, 6))
for i in range(X_standardized.shape[1]):  # Boucler sur toutes les colonnes de X_standardized
    plt.scatter(range(len(X_standardized)), X_standardized[:, i], label=X.columns[i], alpha=0.5)
plt.title('Nuage de points pour les variables standardisées')
plt.xlabel('Index de l\'échantillon')
plt.ylabel('Valeur de la variable standardisée')
plt.legend()
plt.show()
```

[![NP_après_standardisation](https://github.com/AnaisNait/Detection-de-fraude-par-carte-de-credit/assets/103700341/b03d7f7e-cfad-4654-9280-fc863cbacdd8)](#)

> On remarque que les données sont centrées autour de zéro après la standardisation car les points sont distribués de manière symétrique par rapport à l'axe y=0.
> En comparant les deux graphiques, on remarque que la dispersion des données a changé après la standardisation tel que les points dans le nuage de points standardisé sont plus regroupés 
 

# 6- Modèles de prédiction:  
### Division des données
On isole la variable classe y afin de s'assurer que le modèle ne « voit » pas les valeurs de la variable cible lors de l'apprentissage
```
from sklearn.model_selection import train_test_split
y = data.Class
```
On divise les données comme suit: 
- X_train : Ensemble d'apprentissage pour les variables explicatives
- X_test : Ensemble de test pour les variables explicatives
- y_train : Ensemble d'apprentissage pour la variable cible
- y_test : Ensemble de test pour la variable cible
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```
### Isolation Forest  
Créer le modèle de détection d'anomalies  
```
from sklearn.ensemble import IsolationForest
model = IsolationForest()  
```  
Entrainer le modèle sur l'ensemble d'apprentissage  
```
model.fit(X_train)    
```
Prédire sur l'ensemble de test   
```
y_pred = model.predict(X_test)  
y_pred[y_pred == 1] = 0  # Prédiction correcte
y_pred[y_pred == -1] = 1  # Anomalie détectée

```
L'algorithme IsolationForest a le principe de diviser les données en deux classes : 1 pour les données normales et -1 pour les anomalies. Dans le contexte de notre problème de détection de fraude, nous avons deux classes 0 et 1. Pour éviter les confusions, on transforme les prédictions pour qu'elles correspondent à la classe réelle (0 pour normal, 1 pour anomalie)

 Évaluer les performances du modèle
```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 
```
                precision  recall  f1-score   support

           0     1.00       0.96      0.98     56864
           1     0.04       0.83      0.07     98

    accuracy                              0.96     56962
    macro avg        0.52       0.90      0.53     56962
    weighted avg     1.00       0.96      0.98     56962


* Précision: La précision pour la classe 0 est très élevée (1.00), ce qui indique que presque toutes les transactions identifiées comme normales le sont réellement. En revanche, la précision pour la classe 1 est très faible (0.04).   
* Rappel (Recall) : Le rappel pour la classe 0 est de 0.96, ce qui signifie que le modèle a correctement identifié la plupart des transactions normales. Pour la classe 1, le rappel est de 0.83, ce qui indique que le modèle a réussi à identifier une proportion importante des transactions frauduleuses.    

bien que le modèle semble performant pour détecter les transactions normales, il a du mal à identifier les transactions frauduleuses. On va donc essayer d'améliorer les performances du modèle pour donner de plus bons résultats.  

### Suréchantillonnage  
Le but est d'augmenter le nombre d'échantillons de la classe minoritaire (fraudes) pour équilibrer les classes. Pour ça on utilise la méthode SMOTE pour créer des échantillons synthétiques de la classe minoritaire.
```
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
``` 
Puis on re-applique le modèle :  
``` 

model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)
y_pred[y_pred == 1] = 0  # Prédiction correcte (normal)
y_pred[y_pred == -1] = 1  # Anomalie détectée
print(classification_report(y_test, y_pred))
```
```  
        0       1.00      0.98      0.99     56864
        1       0.03      0.28      0.05        98

    accuracy                        0.98     56962
   macro avg    0.51      0.63      0.52     56962
weighted avg    1.00      0.98      0.99     56962
``` 
On remarque une légère amélioration par rapport aux résultats précédents. 

### SVM 
On fait crée le modèle, puis on l'entraine sur sur les données d'entraînement pour enfin faire des prédictions sur les données de test comme suit: 
```
from sklearn.svm import SVC
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
```
Affichons les résultats du modèle:
```
print(classification_report(y_test, y_pred_svm))
```
```
            precision    recall  f1-score   support

        0     1.00      1.00      1.00     56864
        1     0.60      0.30      0.40     98

    accuracy                      1.00     56962
    macro avg    0.80   0.65      0.70     56962
    weighted avg 1.00   1.00      1.00     56962
```
Pour la classe 0 (transactions non frauduleuses), le modèle a une précision, un rappel et un score F1 de 1. Cela signifie que le modèle identifie correctement toutes les transactions non frauduleuses et ne génère pas de faux positifs.

Pour la classe 1 (transactions frauduleuses), la précision est de 0.60, le rappel de 0.30 et le score F1 de 0.40. Cela indique que le modèle a du mal à identifier toutes les transactions frauduleuses. Il identfie donc que 40% des transactions frauduleuses.

En comparant ces résultats avec ceux de l'Isolation Forest, il semble que le SVM ait de meilleures performances.   

# 7- Comparaison des résultats  
| Algorithme             | Précision Classe 0 | Rappel Classe 0 | Précision Classe 1 | Rappel Classe 1 |
|------------------------|--------------------|-----------------|--------------------|-----------------|
| Random Forest          | 1.00               | 1.00            | 0.93               | 0.76            |
| Détection d'anomalies  | 1.00               | 0.98            | 0.03               | 0.28            |
| SVM                    | 1.00               | 1.00            | 0.60               | 0.30            |  

_Random Forest :_  
- Offre une excellente précision et rappel pour la classe 0.  
- Offre une très bonne précision mais un rappel moyen pour la classe 1, ce qui signifie qu'il est bon pour détecter les fraudes, mais il manque encore quelques transactions frauduleuses.   
  
_Détection d'anomalies :_   
- Offre une précision élevée pour la classe 0, mais son rappel pour la classe 1 est très faible, ce qui signifie qu'il détecte très peu de fraudes (beaucoup de faux négatifs).  
  
_SVM :_  
- Offre une excellente précision et rappel pour la classe 0.
La précision et le rappel pour la classe 1 sont inférieurs à ceux de Random Forest, mais meilleurs que ceux de la détection d'anomalies.  

[![Comparaison](https://github.com/AnaisNait/Detection-de-fraude-par-carte-de-credit/assets/103700341/8787bda5-569b-4198-84ca-df27263987df)](#)

Random Forest et SVM montrent des performances solides pour la classe majoritaire (classe 0) et de bonnes performances pour la classe minoritaire (classe 1), avec Random Forest offrant un meilleur rappel pour la classe 1.  
Détection d'anomalies est moins efficace pour détecter les fraudes, avec un rappel et une précision faibles pour la classe 1.  

# 8- Conclusion
L'algorithme de détection d'anomalies  a montré une performance plutôt médiocre pour la détection de fraudes dans ce dataset. Bien qu'il ait réussi à détecter la majorité des transactions non frauduleuses avec une précision élevée (98%), il a eu du mal à identifier les transactions frauduleuses, comme en témoigne le rappel très faible pour cette classe (83%).   
Le SVM a affiché une meilleure performance globale par rapport à l'Isolation Forest. Il a réussi à détecter les transactions non frauduleuses avec une précision parfaite de 1,00 et un rappel de 0,96. Cependant, comme l'Isolation Forest, il a eu du mal à identifier les transactions frauduleuses, avec une précision de seulement 0,04 et un rappel de 0,83.     
En conclusion, notre problème peut être résolu grâce à plusieurs modèles qui possèdent tous des précsions équivalentes: Isolation Forest et SVM. Malgrè le deséquilibre dans notre dataset, les résultats sont bons. En revanche, la méthode de sur-échantillonage n'améliore pas la performance des algorithmes.




### Bibliographie et références
https://www.kaggle.com/code/laurajezequel/credit-card-fraud-detection  
https://www.kaggle.com/code/imanelmountasser/d-tection-de-fraude/notebook  
https://www.kaggle.com/code/julienchoukroun99/d-tection-de-fraude-de-carte-bancaire/notebook
https://www.youtube.com/watch?v=FTtzd31IAOw&t=1206s








