# Projet de Classification d'Images (CNN, ResNet50, MobileNet)

Ce projet entraîne et compare plusieurs modèles de deep learning (CNN personnalisé, ResNet50, MobileNet) pour la classification d'images, avec une interface web Streamlit pour tester les modèles.

## Prérequis

- **Python 3.10+**

## Installation

### 1. Décompresser les données

Avant toute chose, dézippez le dossier `data` à la racine du projet :

```bash
unzip data.zip
```

### 2. Installer les dépendances

```bash
pip install torch torchvision scikit-learn matplotlib numpy pillow streamlit
```

## Démarrage du projet

Lancez les commandes suivantes dans l'ordre :

### 1. Entraînement des modèles

```bash
python src/entrainement_cnn.py
python src/entrainement_resnet50.py
python src/entrainement_mobilenet.py
```

### 2. Évaluation des modèles

```bash
python src/evaluation.py
```

### 3. Lancement de l'application web

```bash
streamlit run src/application.py
```

L'application s'ouvrira automatiquement dans votre navigateur (par défaut sur `http://localhost:8501`).

## Structure du projet

```
.
├── data/                          # Données (à dézipper)
├── src/
│   ├── entrainement_cnn.py        # Entraînement du CNN personnalisé
│   ├── entrainement_resnet50.py   # Entraînement du ResNet50
│   ├── entrainement_mobilenet.py  # Entraînement du MobileNet
│   ├── evaluation.py              # Évaluation et comparaison des modèles
│   └── application.py             # Application web Streamlit
└── README.md
```
