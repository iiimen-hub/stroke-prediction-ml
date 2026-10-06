# Prédiction des AVC par Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iiimen-hub/stroke-prediction-ml/blob/main/stroke_prediction.ipynb)

Projet réalisé dans le cadre du cours *Atelier Systèmes Intelligents 2* (Master DSIR 1, ISET Sfax, 2025-2026).

L'objectif est de comparer plusieurs modèles de classification pour prédire le risque d'accident vasculaire cérébral (AVC) à partir de données médicales et socio-démographiques.

## Dataset

- **Source** : [Stroke Prediction Dataset (Kaggle, fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- 5 110 patients, 11 variables explicatives + la cible `stroke` (0 = pas d'AVC, 1 = AVC)
- **Classes fortement déséquilibrées** : 95,1 % sans AVC, 4,9 % avec AVC
- Le fichier CSV n'est pas inclus : voir [`data/README.md`](data/README.md)

## Méthodologie

1. **Prétraitement** : suppression de `id`, traitement des valeurs manquantes de `bmi` (médiane), encodage (Label Encoding pour les variables binaires, One-Hot pour `work_type` et `smoking_status`), normalisation (StandardScaler), split train/test 80/20 stratifié
2. **Analyse exploratoire (EDA)** : distributions, corrélations, boxplots, variables catégorielles vs cible
3. **Modèles comparés** (validation croisée stratifiée 10-fold) : Régression Logistique, KNN, SVM, Arbre de Décision, Forêt Aléatoire, Bagging, AdaBoost, Gradient Boosting
4. **Évaluation** sur le jeu de test : Accuracy, Précision, Rappel, F1-score, AUC-ROC, courbes ROC, matrice de confusion
5. **Optimisation** des hyperparamètres avec GridSearchCV (Gradient Boosting, Random Forest)

## Résultats

Avec des classes aussi déséquilibrées, l'accuracy (~95 %) est trompeuse : un modèle qui prédit toujours « pas d'AVC » obtient déjà 95,1 %. Les modèles sont donc comparés principalement sur l'**AUC-ROC**.

| Modèle (jeu de test) | AUC-ROC |
|---|---|
| Régression Logistique | 0,839 |
| Random Forest (optimisé) | 0,837 |
| AdaBoost | 0,826 |
| Gradient Boosting (optimisé) | 0,826 |

Limite principale : le **rappel sur la classe AVC reste très faible** (≈ 0,02 pour les meilleurs modèles). Les modèles classent correctement les patients selon leur risque (bon AUC), mais détectent presque aucun cas positif avec le seuil de décision par défaut.

## Pistes d'amélioration

- Rééquilibrage des classes (SMOTE, `class_weight='balanced'`)
- Ajustement du seuil de décision pour privilégier le rappel
- Évaluation avec des métriques adaptées (rappel, PR-AUC)

## Technologies

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Google Colab

## Contenu du dépôt

```
stroke_prediction.ipynb     # notebook complet (code + résultats)
docs/                       # rapport (.docx) et présentation (.pptx)
data/README.md              # instructions pour télécharger le dataset
requirements.txt
```

## Exécution

**Sur Google Colab** : cliquer sur le badge « Open in Colab », puis importer `healthcare-dataset-stroke-data.csv` dans l'environnement.

**En local** :

```bash
pip install -r requirements.txt
jupyter notebook stroke_prediction.ipynb
```

## Auteures

Imen Fatnassi 
