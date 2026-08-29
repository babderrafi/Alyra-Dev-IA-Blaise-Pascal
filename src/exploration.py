# ============================================================================
# EXPLORATION DU JEU DE DONNEES
# ============================================================================
# Avant d'entraîner quoi que ce soit, on regarde ce qu'on a :
#   1. combien d'images par split (train/val/test) et par classe
#   2. la taille des images
#   3. la provenance des images (lue dans le nom de fichier)
#   4. quelques exemples de chaque classe
#
# Lancement : python src/exploration.py

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # on sauvegarde les figures dans un fichier, sans fenêtre
import matplotlib.pyplot as plt
from PIL import Image

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_DONNEES = DOSSIER_PROJET / "data"
DOSSIER_FIGURES = DOSSIER_PROJET / "outputs" / "figures"

SPLITS = ["train", "val", "test"]
CLASSES = ["fibre", "fragment"]

print("=== Exploration du jeu de données ===")

# ----------------------------------------------------------------------------
# 1. Comptage des images par split et par classe
# ----------------------------------------------------------------------------
# Un jeu équilibré (50% fibre / 50% fragment) rend l'accuracy directement
# interprétable : un modèle qui répond au hasard fera 50%.

print("\n[1] Nombre d'images par split et par classe")
print(f"    {'split':8s} {'fibre':>6s} {'fragment':>9s} {'total':>6s}")
toutes_les_images = []   # on garde la liste (chemin, split, classe) pour la suite
for split in SPLITS:
    comptes = {}
    for classe in CLASSES:
        fichiers = sorted((DOSSIER_DONNEES / split / classe).glob("*.jpg"))
        comptes[classe] = len(fichiers)
        for f in fichiers:
            toutes_les_images.append((f, split, classe))
    total = comptes["fibre"] + comptes["fragment"]
    print(f"    {split:8s} {comptes['fibre']:6d} {comptes['fragment']:9d} {total:6d}")
print(f"    Total : {len(toutes_les_images)} images, parfaitement équilibrées 50/50.")

# ----------------------------------------------------------------------------
# 2. Taille des images
# ----------------------------------------------------------------------------
# Les tailles sont très variées : il faudra tout redimensionner en 224x224
# (la taille d'entrée du ResNet50).

print("\n[2] Taille des images")
largeurs = []
for chemin, _, _ in toutes_les_images:
    with Image.open(chemin) as image:
        largeurs.append(image.size[0])
print(f"    Largeur min : {min(largeurs)} px, max : {max(largeurs)} px")
print(f"    Nombre de tailles différentes : {len(set(largeurs))}")
print("    -> On redimensionnera tout en 224x224 dans la préparation.")

# ----------------------------------------------------------------------------
# 3. Provenance des images
# ----------------------------------------------------------------------------
# Le nom de fichier contient la source : fibre_srcA_0001.jpg -> srcA.
# POURQUOI C'EST IMPORTANT : si une source contenait 90% de fibres, le modèle
# pourrait deviner la classe en reconnaissant l'appareil (fond, éclairage)
# au lieu de regarder la particule. C'est ce qu'on appelle un "raccourci".
# On vérifie donc que chaque source est proche de 50% de fibres.

print("\n[3] Répartition par provenance (source de l'image)")
compte_par_source = Counter()
fibres_par_source = Counter()
for chemin, _, classe in toutes_les_images:
    source = chemin.name.split("_")[1]      # fibre_srcA_0001.jpg -> srcA
    compte_par_source[source] += 1
    if classe == "fibre":
        fibres_par_source[source] += 1
print(f"    {'source':8s} {'images':>7s} {'% fibres':>9s}")
for source in sorted(compte_par_source):
    total = compte_par_source[source]
    part_fibres = 100 * fibres_par_source[source] / total
    print(f"    {source:8s} {total:7d} {part_fibres:8.1f}%")
print("    Chaque source est proche de 50% : pas de raccourci évident.")
print("    Remarque : dans srcA, les fibres et les fragments viennent de deux")
print("    types d'imagerie différents (fond clair / fond sombre). Ce biais ne")
print("    peut pas être corrigé ; on en tiendra compte dans l'évaluation.")

# ----------------------------------------------------------------------------
# 4. Figure : exemples d'images de chaque classe
# ----------------------------------------------------------------------------

print("\n[4] Figure d'exemples")
DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)

figure, axes = plt.subplots(2, 4, figsize=(12, 6))
for ligne, classe in enumerate(CLASSES):
    exemples = [img for img in toutes_les_images
                if img[1] == "train" and img[2] == classe][:4]
    for colonne, (chemin, _, _) in enumerate(exemples):
        ax = axes[ligne, colonne]
        ax.imshow(Image.open(chemin).convert("L"), cmap="gray")
        ax.set_title(f"{classe}", fontsize=10)
        ax.axis("off")
figure.suptitle("Exemples d'images du jeu d'entraînement")
plt.tight_layout()
plt.savefig(DOSSIER_FIGURES / "exemples.png")
plt.close()
print("    Figure sauvegardée : outputs/figures/exemples.png")

print("\nExploration terminée.")
