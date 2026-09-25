# Aegis Insurance - Prédiction des frais médicaux

Projet réalisé dans le cadre de la formation **Data Analyst de Simplon Lyon**, brief « Machine Learning & Régression ».

**Application en ligne : [aegis-insurance-zohair-brice.streamlit.app](https://aegis-insurance-zohair-brice.streamlit.app/)**

## Contexte

Aegis Insurance est une compagnie d'assurance santé fictive. L'objectif est de prédire les frais médicaux annuels d'un client à partir de son profil, puis de mettre ce modèle à disposition dans une application de simulation.

## Données

Fichier `insurance-data.csv` : 1 338 clients (1 337 après suppression d'un doublon) et 7 colonnes.

| Colonne | Description |
|---|---|
| `age` | Âge du client |
| `sex` | Sexe |
| `bmi` | Indice de masse corporelle (IMC) |
| `children` | Nombre d'enfants à charge |
| `smoker` | Fumeur (yes / no) |
| `region` | Région de résidence (4 régions) |
| `expenses` | Frais médicaux annuels en $ (**variable à prédire**) |

## Démarche

1. **Analyse exploratoire** : distributions, lien entre les frais et le tabac, l'IMC et l'âge, valeurs atypiques.
2. **Prétraitement** : suppression du doublon, encodage des variables catégorielles, séparation train / test (80 % / 20 %), standardisation.
3. **Modélisation** : entraînement de 4 algorithmes de régression, puis optimisation des hyperparamètres avec GridSearchCV.
4. **Comparaison** des modèles sur le même jeu de test et choix du modèle final.
5. **Application Streamlit** : exploration des données et simulateur de frais.

## Résultats

Résultats sur le jeu de test (section 9 du notebook) :

| Modèle | MAE ($) | MSE | R² |
|---|---|---|---|
| Régression Linéaire | 4 177 | 35 481 472 | 0,807 |
| KNN | 3 472 | 29 900 869 | 0,837 |
| Arbre de Décision | 2 917 | 38 918 076 | 0,788 |
| Random Forest | 2 665 | 22 227 850 | 0,879 |
| **Random Forest optimisé** | **2 440** | **18 137 957** | **0,901** |

**Modèle retenu : Random Forest optimisé** (`n_estimators=200`, `max_depth=5`), avec le R² le plus élevé et la MAE et la MSE les plus faibles.

## L'application

L'application se présente comme un parcours en 4 étapes :

1. **Le constat** : chiffres clés et comparaison fumeurs / non-fumeurs.
2. **Les facteurs de risque** : frais selon l'IMC et l'âge, avec filtre par région.
3. **Le choix du modèle** : comparaison des 5 modèles et importance des variables.
4. **Le simulateur** : estimation des frais d'un client, position parmi les autres clients et scénarios « Et si… ? ».

Les graphiques sont interactifs (survol, zoom, filtre par légende).

## Structure du repo

```
├── assurance_Zohair_Brice.ipynb   # Notebook : analyse, modèles, comparaison, export du modèle
├── insurance-data.csv             # Données
├── modele_aegis.joblib            # Modèle final + scaler + ordre des colonnes (section 10 du notebook)
├── app.py                         # Application Streamlit
├── requirements.txt               # Bibliothèques nécessaires
└── .streamlit/config.toml         # Couleurs de l'application
```

## Lancer l'application en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Outils

Python, pandas, scikit-learn, seaborn, matplotlib, Plotly, Streamlit, joblib.

## Auteurs

Projet réalisé en binôme par **Zohair Nazhaoui** et **Brice**.
