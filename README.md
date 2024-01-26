# Détection de fraude par carte de crédit
Ce projet vise à analyser et visualiser un dataset de transactions par carte de crédit afin d'identifier les transactions frauduleuses parmi celles-ci et ceci en utilisant un algorithme de machine learning. 

##1- Introduction  
Le secteur financier est confronté à des défis perpétuels liés à la sécurité des transactions, en particulier dans le contexte de la montée en puissance de la numérisation des services bancaires. La fraude par carte de crédit demeure une menace persistante, nécessitant des solutions innovantes pour préserver l'intégrité des transactions et la confiance des utilisateurs. Dans cette optique, ce projet traite un ensemble de données de transactions par carte de crédit afin de détecter les transactions frauduleuses parmi celles-ci, en exploitant les capacités de l'apprentissage automatique.
#Contexte 
La fraude bancaire a connu une évolution remarquable au fil des décennies. Des méthodes traditionnelles, telles que les contrefaçons de chèques et les vols de cartes, ont cédé la place à des pratiques plus sophistiquées dans le monde numérique. L'émergence de la cybercriminalité a donné naissance à des formes de fraude complexes, exploitant les failles des systèmes pour dérober des informations sensibles.  
La fraude par carte de crédit demeure l'une des manifestations les plus répandues et pernicieuses de la fraude financière. Des attaques telles que le clonage de cartes, les achats non autorisés et le vol d'informations personnelles persistent, malgré les mesures de sécurité en place. L'utilisation croissante des services bancaires en ligne et des transactions électroniques expose davantage les utilisateurs à ces risques.  

##2- État de l'Art de la Fraude Bancaire
#Antiquité à Moyen Âge :
Les premières formes de fraudes financières étaient souvent liées à la contrefaçon de pièces de monnaie et à la falsification de documents.[1]  
#19ème siècle :
L'introduction des chèques a donné lieu à des fraudes telles que la falsification de signatures.
Les banques ont commencé à embaucher des experts en écriture pour vérifier l'authenticité des signatures.[2]  
Cependant, cette approche était limitée par la vitesse de traitement et la subjectivité des évaluations.  

#Début du 20ème siècle :  
L'avènement des transactions électroniques a introduit de nouveaux défis. Les premières formes de fraude par cartes de crédit ont émergé.  
Les banques ont commencé à utiliser des méthodes statistiques pour détecter des modèles inhabituels dans les transactions.[3]  

#Années 1960-1970 :  
L'essor des ordinateurs a ouvert la voie à automatiser de nombreuses opérations. Cela a permis d'améliorer l'efficacité, mais a également créé de nouvelles vulnérabilités à exploiter.  
Les banques ont commencé à utiliser des systèmes de détection automatisés pour surveiller les activités les activités financières suspectes. Cela a souvent été basé sur des règles prédéfinies et des modèles statistiques.[4]  
les modèles basés sur la régression logistique étaient parmi les approches les plus utilisées pour identifier les transactions potentiellement frauduleuses. Ces modèles pouvaient prendre en compte divers paramètres, tels que le montant, l'emplacement, l'heure, etc., et attribuer des scores de probabilité à ces transactions. Cependant, ils étaient limités dans leur capacité à capturer des schémas de fraude complexes, en particulier lorsque les relations entre les variables sont non linéaires.   

#Années 1990 à aujourd'hui :  
L'avènement d'Internet a conduit à des fraudes en ligne, telles que le vol d'identité et la fraude par phishing.
Les banques ont développé des outils de détection avancés basés sur l'intelligence artificielle et l'apprentissage automatique pour détecter les schémas frauduleux.[5]  

