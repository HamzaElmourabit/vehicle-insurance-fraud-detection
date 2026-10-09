# 🚗 Vehicle Insurance Fraud Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Ce projet met en œuvre des modèles de **Machine Learning** pour identifier et prédire les réclamations d'assurance automobile frauduleuses. L'objectif principal est de réduire les pertes financières des compagnies d'assurance tout en accélérant le traitement des demandes légitimes.

---

## 📌 Table des matières
- [Aperçu du Projet](#-aperçu-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Jeu de Données](#-jeu-de-données)
- [Architecture & Pipeline ML](#-architecture--pipeline-ml)
- [Structure du Projet](#-structure-du-projet)
- [Installation et Configuration](#-installation-et-configuration)
- [Utilisation](#-utilisation)
- [Résultats & Performances](#-résultats--performances)
- [Technologies Utilisées](#-technologies-utilisées)
- [Auteur](#-auteur)

---

## 🔎 Aperçu du Projet

La fraude à l'assurance automobile représente des milliards de dollars de pertes chaque année. Ce projet résout ce problème sous la forme d'une **classification binaire** (`0 = Réclamation légitime`, `1 = Réclamation frauduleuse`). 

Les données d'assurance étant fortement déséquilibrées (très peu de cas de fraude par rapport aux réclamations normales), ce projet intègre des techniques avancées de rééchantillonnage (ex: **SMOTE**) et d'optimisation des hyperparamètres.

---

## ✨ Fonctionnalités

* **Analyse Exploratoire des Données (EDA) :** Visualisation des tendances, de la répartition géographique, des types de véhicules et du profil des assurés.
* **Prétraitement & Feature Engineering :** Gestion des valeurs manquantes, encodage des variables catégorielles (One-Hot / Target Encoding) et normalisation.
* **Gestion du Déséquilibre de Classes :** Utilisation de SMOTE / Class Weighting pour éviter le biais en faveur des classes majoritaires.
* **Modélisation Multi-Algorithmes :** Entraînement et comparaison de plusieurs modèles (Régression Logistique, Random Forest, XGBoost, LightGBM).
* **Évaluation Métrique Métier :** Focus sur le **Rappel (Recall)** et le **ROC-AUC** afin de limiter au maximum les Faux Négatifs (fraudes manquées).

---

## 📊 Jeu de Données

Le dataset contient des informations historiques sur les demandes d'indemnisation. Les caractéristiques clés incluent :
* **Informations sur le conducteur/assuré :** Âge, genre, statut marital, historique de réclamations.
* **Informations sur le véhicule :** Catégorie du véhicule, prix, âge du véhicule.
* **Détails du sinistre :** Type de collision, présence de témoins, rapport de police, franchise (deductible).
* **Variable Cible (`FraudFound_P`) :** `1` si la demande est frauduleuse, `0` sinon.

---

## ⚙️ Architecture & Pipeline ML

```mermaid
graph TD;
    A[Données Brutes] --> B[Nettoyage & EDA];
    B --> C[Feature Engineering & Scaling];
    C --> D[Rééquilibrage des classes - SMOTE];
    D --> E[Entraînement des Modèles - XGBoost/RandomForest];
    E --> F[Évaluation des performances];
    F --> G[Prédiction / Déploiement];
```

---

## 📁 Structure du Projet

```text
vehicle-insurance-fraud-detection/
│
├── data/
│   ├── raw/                  # Données brutes
│   └── processed/            # Données nettoyées et préparées
│
├── notebooks/
│   ├── 01_eda.ipynb          # Analyse exploratoire
│   └── 02_modeling.ipynb     # Prétraitement, entraînement et évaluation
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Pipeline de transformation de données
│   ├── train.py              # Script d'entraînement des modèles
│   └── predict.py            # Script d'inférence / prédiction
│
├── app.py                    # Application Web (Streamlit / Flask) pour les prédictions
├── requirements.txt          # Dépendances Python
├── .gitignore                # Fichiers à ignorer par Git
└── README.md                 # Documentation du projet
```

---

## 🚀 Installation et Configuration

### Prérequis
* Python 3.8+
* `pip` ou `conda`

### Étapes d'installation

1. **Cloner le dépôt Git :**
   ```bash
   git clone https://github.com/HamzaElmourabit/vehicle-insurance-fraud-detection.git
   cd vehicle-insurance-fraud-detection
   ```

2. **Créer et activer un environnement virtuel :**
   ```bash
   # Sur Linux / macOS
   python3 -m venv venv
   source venv/bin/activate

   # Sur Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Utilisation

### 1. Exécution de l'entraînement
Pour lancer le pipeline complet de préparation des données et d'entraînement du modèle :
```bash
python src/train.py
```

### 2. Lancement de l'interface graphique (optionnel)
Si une application **Streamlit** est disponible pour tester les prédictions en direct :
```bash
streamlit run app.py
```

---

## 📈 Résultats & Performances

Les modèles ont été évalués principalement sur le **Recall** (sensibilité) pour maximiser la détection des fraudes et sur l'**AUC-ROC** :

| Modèle | Précision (Precision) | Rappel (Recall) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Régression Logistique** | 0.42 | 0.85 | 0.56 | 0.78 |
| **Random Forest** | 0.65 | 0.78 | 0.71 | 0.84 |
| **XGBoost (Optimal)** | **0.72** | **0.88** | **0.79** | **0.89** |

---

## 🛠️ Technologies Utilisées

* **Langage :** Python
* **Analyse de Données :** Pandas, NumPy
* **Visualisation :** Matplotlib, Seaborn
* **Machine Learning :** Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn
* **Déploiement / Interface :** Streamlit / Flask

---

## 👤 Auteur

**Hamza El Mourabit**
* GitHub : [@HamzaElmourabit](https://github.com/HamzaElmourabit)
* LinkedIn : [Hamza El Mourabit](https://linkedin.com/in/)

---
⭐ *N'hésitez pas à laisser une étoile si ce projet vous a été utile !*
