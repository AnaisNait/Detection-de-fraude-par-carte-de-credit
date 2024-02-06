# Détection de fraude par carte de crédit
Ce projet vise à analyser et visualiser un dataset de transactions par carte de crédit afin d'identifier les transactions frauduleuses parmi celles-ci et ceci en utilisant un algorithme de machine learning. 

# 1- Introduction  
Le secteur financier est confronté à des défis perpétuels liés à la sécurité des transactions, en particulier dans le contexte de la montée en puissance de la numérisation des services bancaires. La fraude par carte de crédit demeure une menace persistante, nécessitant des solutions innovantes pour préserver l'intégrité des transactions et la confiance des utilisateurs. Dans cette optique, ce projet traite un ensemble de données de transactions par carte de crédit afin de détecter les transactions frauduleuses parmi celles-ci, en exploitant les capacités de l'apprentissage automatique.  
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
L'essor des ordinateurs a ouvert la voie à automatiser de nombreuses opérations. Cela a permis d'améliorer l'efficacité, mais a également créé de nouvelles vulnérabilités à exploiter.   
Les banques ont commencé à utiliser des systèmes de détection automatisés pour surveiller les activités financières suspectes. Cela a souvent été basé sur des règles prédéfinies et des modèles statistiques.  
les modèles basés sur la régression logistique étaient parmi les approches les plus utilisées pour identifier les transactions potentiellement frauduleuses. Ces modèles pouvaient prendre en compte divers paramètres, tels que le montant, l'emplacement, l'heure, etc., et attribuer des scores de probabilité à ces transactions. Cependant, ils étaient limités dans leur capacité à capturer des schémas de fraude complexes, en particulier lorsque les relations entre les variables sont non linéaires.   
### Années 1990 à aujourd'hui :    
L'avènement d'Internet a conduit à des fraudes en ligne, telles que le vol d'identité et la fraude par phishing.
Les banques ont développé des outils de détection avancés basés sur l'intelligence artificielle et l'apprentissage automatique pour détecter les schémas frauduleux.  

#### _Méthodes d'apprentissage automatique:_  
* Machines à Vecteurs de Support (SVM):  
Elles permettent de créer des frontières de décision non linéaires. Elles sont efficaces pour traiter des ensembles de données où les classes ne sont pas linéairement séparables.  
* Random Forest:  
Les SVM peuvent être utilisées pour séparer les transactions frauduleuses des transactions légitimes en construisant des hyperplans dans un espace multidimensionnel séparant les transactions en deux classes.  
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
### Analyse statistique univariée    
   Time            V1            V2            V3            V4  \                              
count  284807.000000  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  
mean    94813.859575  1.168375e-15  3.416908e-16 -1.379537e-15  2.074095e-15   
std     47488.145955  1.958696e+00  1.651309e+00  1.516255e+00  1.415869e+00     
min         0.000000 -5.640751e+01 -7.271573e+01 -4.832559e+01 -5.683171e+00     
25%     54201.500000 -9.203734e-01 -5.985499e-01 -8.903648e-01 -8.486401e-01     
50%     84692.000000  1.810880e-02  6.548556e-02  1.798463e-01 -1.984653e-02     
75%    139320.500000  1.315642e+00  8.037239e-01  1.027196e+00  7.433413e-01     
max    172792.000000  2.454930e+00  2.205773e+01  9.382558e+00  1.687534e+01     

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

                V25           V26           V27           V28         Amount  \  
count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  284807.000000     
mean   5.340915e-16  1.683437e-15 -3.660091e-16 -1.227390e-16      88.349619     
std    5.212781e-01  4.822270e-01  4.036325e-01  3.300833e-01     250.120109     
min   -1.029540e+01 -2.604551e+00 -2.256568e+01 -1.543008e+01       0.000000     
25%   -3.171451e-01 -3.269839e-01 -7.083953e-02 -5.295979e-02       5.600000     
50%    1.659350e-02 -5.213911e-02  1.342146e-03  1.124383e-02      22.000000     
75%    3.507156e-01  2.409522e-01  9.104512e-02  7.827995e-02      77.165000     
max    7.519589e+00  3.517346e+00  3.161220e+01  3.384781e+01   25691.160000     

               Class    
count  284807.000000    
mean        0.001727    
std         0.041527    ()
min         0.000000    
25%         0.000000    
50%         0.000000    
75%         0.000000    
max         1.000000    
### Prétraitement des données  
Le code ci-dessous, permet de dessiner un boxplot pour identifier les colonnes du Dataset qui contiennent des valeurs aberrantes.   
Il n'y a pas de points situés en dehors de la boite à moustaches. Tous les points sont à l'intérieur, donc il n'y a pas de valeurs aberrantes.  
### Standardisation  
Dans cet exemple, data_scaled[:, 0] et data_scaled[:, 1] représentent les deux premières colonnes standardisées de vos données. Vous devriez ajuster ces indices en fonction des colonnes que vous souhaitez visualiser.

### Bibliographie et références
https://www.kaggle.com/code/laurajezequel/credit-card-fraud-detection  
https://www.kaggle.com/code/imanelmountasser/d-tection-de-fraude/notebook  
https://www.kaggle.com/code/julienchoukroun99/d-tection-de-fraude-de-carte-bancaire/notebook







