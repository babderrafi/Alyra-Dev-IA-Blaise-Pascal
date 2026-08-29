# ============================================================================
# EVALUATION FINALE SUR LE JEU DE TEST
# ============================================================================
# Règle d'or respectée ici : le jeu de TEST n'a servi à AUCUN choix pendant
# l'entraînement (ni arrêt, ni sélection de modèle : tout a été décidé sur
# la validation). Le score mesuré ici est donc un chiffre honnête.
#
# A lancer APRES les entraînements : python src/evaluation.py

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from preparation import charger_donnees
# On réutilise les fonctions des scripts d'entraînement : cela garantit que
# l'on reconstruit EXACTEMENT les mêmes architectures avant de charger les poids.
from entrainement_cnn import creer_cnn
from entrainement_resnet50 import creer_resnet50
from entrainement_mobilenet import creer_mobilenet

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_SORTIES = DOSSIER_PROJET / "outputs"
DOSSIER_FIGURES = DOSSIER_SORTIES / "figures"

print("=== Evaluation sur le jeu de TEST ===")
print("(ce jeu n'a servi à aucun choix pendant l'entraînement)\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)

# ----- 1. Données de test -----
_, _, chargeur_test, classes = charger_donnees()
print(f"{len(chargeur_test.dataset)} images de test, classes : {classes}")

# La provenance de chaque image de test, lue dans le nom de fichier
# (fibre_srcA_0001.jpg -> srcA). Le chargeur de test ne mélange pas les
# images, donc l'ordre des prédictions suivra cet ordre.
sources = [Path(chemin).name.split("_")[1]
           for chemin, _ in chargeur_test.dataset.samples]

# ----- 2. Les modèles à évaluer -----
modeles_a_evaluer = [
    ("CNN simple", creer_cnn, DOSSIER_SORTIES / "cnn.pth"),
    ("ResNet50", creer_resnet50, DOSSIER_SORTIES / "resnet50.pth"),
    ("MobileNetV3", creer_mobilenet, DOSSIER_SORTIES / "mobilenet.pth"),
]

resultats = []   # on garde les scores pour la comparaison finale

for nom, creer_modele, chemin_poids in modeles_a_evaluer:
    if not chemin_poids.is_file():
        print(f"\n{nom} : pas de modèle sauvegardé ({chemin_poids.name} absent), ignoré.")
        continue

    # ----- 3. Chargement du modèle entraîné -----
    modele = creer_modele()
    modele.load_state_dict(torch.load(chemin_poids, map_location="cpu"))
    modele.to(device)
    modele.eval()   # indispensable : fige Dropout et BatchNorm

    # ----- 4. Prédictions sur tout le jeu de test -----
    # y_true = les vraies classes, y_pred = les prédictions du modèle
    # (les noms standards de scikit-learn)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, etiquettes in chargeur_test:
            sorties = modele(images.to(device))
            y_pred += sorties.argmax(dim=1).cpu().tolist()
            y_true += etiquettes.tolist()

    # ----- 5. Métriques -----
    accuracy = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 60)
    print(f"{nom} — accuracy sur le test : {accuracy:.3f}")
    print("=" * 60)
    # Le rapport détaille précision, rappel et F1 pour chaque classe :
    # avec seulement 84 images de test, l'accuracy seule ne suffit pas.
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    # ----- 6. Contrôle anti-raccourci : score par provenance -----
    # Si le modèle avait appris à reconnaître l'appareil photo plutôt que la
    # particule, ses scores varieraient beaucoup d'une source à l'autre.
    print("  Accuracy par provenance :")
    for source in sorted(set(sources)):
        indices = [i for i, s in enumerate(sources) if s == source]
        bonnes = sum(1 for i in indices if y_pred[i] == y_true[i])
        print(f"    {source:6s} : {bonnes / len(indices):.3f}  ({len(indices)} images)")
    print("  Remarque : srcA garde un biais de fond (fibres sur fond clair,")
    print("  fragments sur fond sombre) impossible à corriger. Le chiffre le")
    print("  plus fiable est donc celui de srcB.")

    # ----- 7. Matrice de confusion -----
    matrice = confusion_matrix(y_true, y_pred)
    affichage = ConfusionMatrixDisplay(matrice, display_labels=classes)
    affichage.plot(cmap="Blues")
    plt.title(f"Matrice de confusion - {nom}")
    nom_fichier = f"confusion_{nom.lower().replace(' ', '_')}.png"
    plt.savefig(DOSSIER_FIGURES / nom_fichier)
    plt.close()
    print(f"  Matrice de confusion : outputs/figures/{nom_fichier}")

    resultats.append((nom, accuracy))

# ----- 8. Comparaison des modèles -----
if len(resultats) >= 2:
    print("\n" + "=" * 60)
    print("COMPARAISON FINALE")
    print("=" * 60)
    for nom, accuracy in resultats:
        print(f"  {nom:12s} : accuracy test = {accuracy:.3f}")
    meilleur = max(resultats, key=lambda r: r[1])
    print(f"\n  Meilleur modèle : {meilleur[0]} ({meilleur[1]:.3f})")
    print("  Attention : avec 84 images de test, un écart de quelques points")
    print("  n'est pas forcément significatif.")

    # Courbes de validation des modèles sur la même figure :
    # on y voit la vitesse de convergence apportée par le transfer learning.
    plt.figure(figsize=(8, 5))
    for nom, fichier in [("CNN simple", "historique_cnn.json"),
                         ("ResNet50", "historique_resnet50.json"),
                         ("MobileNetV3", "historique_mobilenet.json")]:
        if not (DOSSIER_SORTIES / fichier).is_file():
            continue
        with open(DOSSIER_SORTIES / fichier, encoding="utf-8") as f:
            historique = json.load(f)
        plt.plot(historique["val_loss"], label=nom)
    plt.xlabel("Epoque")
    plt.ylabel("Perte de validation (val_loss)")
    plt.title("Vitesse de convergence : de zéro vs transfer learning")
    plt.legend()
    plt.grid(True)
    plt.savefig(DOSSIER_FIGURES / "comparaison_convergence.png")
    plt.close()
    print("\n  Figure : outputs/figures/comparaison_convergence.png")

print("\nEvaluation terminée.")
print("Etape suivante (démo) : streamlit run src/application.py")
