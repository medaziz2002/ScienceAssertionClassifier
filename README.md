📌 Projet Machine Learning - Classification d'Assertions Scientifiques sur Twitter
Cours : HAI817 (2024/2025)
Encadrants : P. Poncelet, K. Todorov, E. Raoufi
Membres du groupe : Mohamed Aziz Belhaj Hssine, Valentin pecqueux, Imen Bouaziz

🎯 Objectif
Développer un modèle de classification supervisée pour catégoriser des tweets selon leur rapport à la science :

{SCI} vs {NON-SCI}

{CLAIM, REF} vs {CONTEXT} (pour les tweets scientifiques)

{CLAIM} vs {REF} vs {CONTEXT} (multi-classes).

📦 Données
Jeu de données SciTweets.

Tweets bruts avec labels hiérarchiques .

Pré-traitement nécessaire : nettoyage (hashtags, emojis, liens), tokenisation, etc.

⚙️ Méthodologie
Pré-traitement :

Suppression des stopwords, lemmatisation, TF-IDF/n-grammes.

Gestion du déséquilibre (upsampling/downsampling).

Modèles testés :

Random Forest, SVM, Naïve Bayes, réseaux de neurones (optionnel).

Évaluation :

Métriques : précision, rappel, F1-score, matrices de confusion.

Sélection de features (importance des variables).
