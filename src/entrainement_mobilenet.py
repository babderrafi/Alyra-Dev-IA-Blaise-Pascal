# ============================================================================
# ENTRAINEMENT DU MOBILENET V3 (transfer learning, modèle léger)
# ============================================================================
# Troisième point de comparaison du projet. MobileNet est un réseau conçu pour
# tourner SUR des appareils légers (téléphone, terrain, embarqué) : ~15 fois
# moins de paramètres que le ResNet50, calculs beaucoup plus rapides.
#
# Question à laquelle il répond : "combien de points de performance paie-t-on
# pour un modèle assez léger pour tourner sur le téléphone d'un opérateur de
# terrain ?" — le scénario d'usage réel de la classification de microplastiques.
#
# Même protocole que les deux autres modèles (même graine, même boucle, même
# early stopping) : c'est la condition pour que la comparaison soit valide.
#
# Lancement : python src/entrainement_mobilenet.py

import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models

from preparation import charger_donnees, SEED

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_SORTIES = DOSSIER_PROJET / "outputs"

# ----- Hyperparamètres (identiques au ResNet50 : comparaison à protocole égal) -----
EPOCHS = 30
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4


def creer_mobilenet():
    """MobileNetV3-Small pré-entraîné sur ImageNet, adapté à nos 2 classes.

    L'idée qui rend MobileNet léger : les convolutions "séparables" — au lieu
    d'un gros filtre qui traite l'espace et les canaux d'un coup, deux petits
    filtres qui les traitent séparément (environ 9 fois moins de calculs).
    """
    modele = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # On gèle tous les poids, comme pour le ResNet50.
    for parametre in modele.parameters():
        parametre.requires_grad = False

    # La tête de MobileNetV3 s'appelle "classifier" (et non "fc") : on remplace
    # sa dernière couche (1024 -> 1000 classes ImageNet) par la nôtre (1024 -> 2).
    # Cette couche neuve est la seule partie entraînée.
    modele.classifier[3] = nn.Linear(1024, 2)
    return modele


def mettre_en_mode_entrainement(modele):
    """Mode entraînement UNIQUEMENT sur la tête, comme pour le ResNet50.

    Le corps gelé reste en mode évaluation (fige aussi les statistiques des
    BatchNorm) ; la tête passe en mode entraînement pour que son Dropout agisse.
    """
    modele.eval()
    modele.classifier.train()


def evaluer(modele, chargeur, loss_fn, device):
    """Calcule la perte moyenne et l'accuracy du modèle sur un jeu de données."""
    modele.eval()
    total_loss = 0.0
    bonnes_reponses = 0
    with torch.no_grad():
        for images, etiquettes in chargeur:
            images, etiquettes = images.to(device), etiquettes.to(device)
            sorties = modele(images)
            total_loss += loss_fn(sorties, etiquettes).item() * len(images)
            predictions = sorties.argmax(dim=1)
            bonnes_reponses += (predictions == etiquettes).sum().item()
    n = len(chargeur.dataset)
    return total_loss / n, bonnes_reponses / n


if __name__ == "__main__":
    print("=== Entraînement du MobileNetV3-Small (transfer learning léger) ===\n")

    # Même graine que les autres modèles : comparaison à conditions égales.
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculs sur : {device}")

    # ----- 1. Données -----
    chargeur_train, chargeur_val, _, classes = charger_donnees()
    print(f"Classes : {classes}")
    print(f"Train : {len(chargeur_train.dataset)} images | "
          f"Val : {len(chargeur_val.dataset)} images "
          f"(le test est réservé à l'évaluation finale)\n")

    # ----- 2. Modèle, perte, optimiseur -----
    modele = creer_mobilenet().to(device)
    total = sum(p.numel() for p in modele.parameters())
    entrainables = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    print(f"Paramètres : {entrainables:,} entraînables sur {total:,} "
          f"({100 * entrainables / total:.2f}%)")
    print(f"(le ResNet50 en compte 23,5 millions : MobileNet est ~15x plus petit)\n")

    loss_fn = nn.CrossEntropyLoss()
    # Seule la nouvelle couche de la tête est confiée à l'optimiseur.
    optimizer = torch.optim.AdamW(modele.classifier[3].parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ----- 3. Boucle d'entraînement (identique aux deux autres modèles) -----
    historique = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_sans_progres = 0
    debut = time.time()

    for epoch in range(EPOCHS):
        # -- Phase d'entraînement : seule la tête apprend --
        mettre_en_mode_entrainement(modele)
        loss_cumulee = 0.0
        for images, etiquettes in chargeur_train:
            images, etiquettes = images.to(device), etiquettes.to(device)
            optimizer.zero_grad()
            loss = loss_fn(modele(images), etiquettes)
            loss.backward()
            optimizer.step()
            loss_cumulee += loss.item() * len(images)
        train_loss = loss_cumulee / len(chargeur_train.dataset)
        scheduler.step()

        # -- Phase de validation --
        val_loss, val_acc = evaluer(modele, chargeur_val, loss_fn, device)

        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)
        historique["val_acc"].append(val_acc)
        print(f"Epoque {epoch + 1:3d}/{EPOCHS} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

        # -- Sélection du meilleur modèle sur la VALIDATION (jamais le test) --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(modele.state_dict())
            epochs_sans_progres = 0
        else:
            epochs_sans_progres += 1
            if epochs_sans_progres >= PATIENCE:
                print(f"\nArrêt anticipé : pas d'amélioration depuis {PATIENCE} époques.")
                break

    duree = time.time() - debut
    print(f"\nEntraînement terminé en {duree / 60:.1f} min "
          f"({epoch + 1} époques). Meilleure val_loss : {best_val_loss:.4f}")

    # ----- 4. Sauvegarde -----
    DOSSIER_SORTIES.mkdir(exist_ok=True)
    torch.save(best_state, DOSSIER_SORTIES / "mobilenet.pth")
    with open(DOSSIER_SORTIES / "historique_mobilenet.json", "w", encoding="utf-8") as f:
        json.dump(historique, f, indent=2)
    print("Modèle sauvegardé : outputs/mobilenet.pth")

    # ----- 5. Courbes d'apprentissage -----
    (DOSSIER_SORTIES / "figures").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(historique["train_loss"], label="perte entraînement (train_loss)")
    plt.plot(historique["val_loss"], label="perte validation (val_loss)")
    plt.xlabel("Epoque")
    plt.ylabel("Perte (entropie croisée)")
    plt.title("Courbes d'apprentissage - MobileNetV3-Small")
    plt.legend()
    plt.grid(True)
    plt.savefig(DOSSIER_SORTIES / "figures" / "courbes_mobilenet.png")
    plt.close()
    print("Courbes sauvegardées : outputs/figures/courbes_mobilenet.png")
    print("\nEtape suivante : python src/evaluation.py")
