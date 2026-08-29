# ============================================================================
# PREPARATION DES DONNEES
# ============================================================================
# Ce fichier prépare les images pour le réseau de neurones :
#   - il définit les transformations appliquées aux images
#   - il crée les trois chargeurs de données : train, val et test
#
# Il est importé par les scripts d'entraînement et d'évaluation, ce qui
# garantit que tout le monde utilise exactement la même préparation.
#
# Test rapide : python src/preparation.py

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----- Réglages principaux du projet -----
DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_DONNEES = DOSSIER_PROJET / "data"
IMG_SIZE = 224     # taille d'entrée du ResNet50
BATCH_SIZE = 32    # nombre d'images traitées à la fois
SEED = 42          # graine : pour que les résultats soient reproductibles

# Moyenne et écart-type des images d'ImageNet : le ResNet50 a été
# pré-entraîné avec ces valeurs, on doit donc normaliser pareil.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def rotation_quart_de_tour(image):
    """Tourne l'image de 0, 90, 180 ou 270 degrés, au hasard.

    On se limite aux quarts de tour : sur une image carrée, ils sont exacts.
    Une rotation de 30 degrés créerait des coins noirs artificiels que le
    réseau pourrait apprendre comme un faux indice.
    """
    k = random.randint(0, 3)
    return image if k == 0 else image.rotate(90 * k)


# Transformations pour l'ENTRAINEMENT, avec augmentation de données :
# on crée des variantes des images (miroir, rotation, luminosité) pour que
# le réseau voie plus de diversité et généralise mieux.
transfo_entrainement = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # Le dataset mélange images couleur et niveaux de gris, et cette
    # différence dépend de la provenance : on force tout en gris pour que le
    # modèle regarde la FORME de la particule, pas le type d'appareil photo.
    # (3 canaux car le ResNet50 attend des images à 3 canaux)
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Lambda(rotation_quart_de_tour),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),               # pixels ramenés entre 0 et 1
    transforms.Normalize(MEAN, STD),
])

# Transformations pour la VALIDATION, le TEST et l'application :
# les mêmes, mais SANS augmentation (on évalue sur les vraies images).
transfo_evaluation = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def charger_donnees():
    """Crée les trois chargeurs de données et renvoie aussi les classes.

    ImageFolder lit les sous-dossiers (fibre/, fragment/) et attribue les
    étiquettes dans l'ordre alphabétique : fibre = 0, fragment = 1.
    """
    jeu_train = datasets.ImageFolder(DOSSIER_DONNEES / "train", transform=transfo_entrainement)
    jeu_val = datasets.ImageFolder(DOSSIER_DONNEES / "val", transform=transfo_evaluation)
    jeu_test = datasets.ImageFolder(DOSSIER_DONNEES / "test", transform=transfo_evaluation)

    # shuffle=True seulement pour l'entraînement : mélanger les images évite
    # que le réseau voie toujours les classes dans le même ordre.
    chargeur_train = DataLoader(jeu_train, batch_size=BATCH_SIZE, shuffle=True)
    chargeur_val = DataLoader(jeu_val, batch_size=BATCH_SIZE)
    chargeur_test = DataLoader(jeu_test, batch_size=BATCH_SIZE)

    return chargeur_train, chargeur_val, chargeur_test, jeu_train.classes


# ----- Petit test de contrôle -----
if __name__ == "__main__":
    print("=== Contrôle de la préparation des données ===\n")
    random.seed(SEED)
    torch.manual_seed(SEED)

    chargeur_train, chargeur_val, chargeur_test, classes = charger_donnees()
    print(f"Classes : {classes}  (fibre = 0, fragment = 1)")
    print(f"Train : {len(chargeur_train.dataset)} images")
    print(f"Val   : {len(chargeur_val.dataset)} images")
    print(f"Test  : {len(chargeur_test.dataset)} images")

    # On vérifie la forme d'un lot : (32 images, 3 canaux, 224, 224)
    images, etiquettes = next(iter(chargeur_train))
    print(f"\nForme d'un lot : {tuple(images.shape)}")
    print(f"Valeurs min/max après normalisation : {images.min():.2f} / {images.max():.2f}")
    print("\nPréparation OK.")
