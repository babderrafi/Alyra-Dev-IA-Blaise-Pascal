# ============================================================================
# ENTRAINEMENT DU RESNET50 (transfer learning)
# ============================================================================
# Principe du transfer learning : le ResNet50 a déjà appris à "voir" (formes,
# textures, contours) sur ImageNet, un jeu de 1,2 million d'images. On
# réutilise ce savoir-faire :
#   - on GELE tout le corps du réseau (ses poids ne bougent plus)
#   - on remplace la dernière couche (1000 classes ImageNet -> nos 2 classes)
#   - on n'entraîne QUE cette nouvelle couche (~4 000 paramètres sur 23,5 M)
# C'est la bonne stratégie avec un petit dataset : très peu de paramètres à
# apprendre, donc très peu de risque de sur-apprentissage.
#
# Lancement : python src/entrainement_resnet50.py

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

# ----- Hyperparamètres -----
# 30 époques suffisent largement : grâce au pré-entraînement, le ResNet50
# converge en quelques époques (le CNN parti de zéro en demande jusqu'à 80).
EPOCHS = 30
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4


def creer_resnet50():
    """ResNet50 pré-entraîné sur ImageNet, adapté à nos 2 classes."""
    modele = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # On gèle tous les poids : le corps du réseau ne sera pas modifié.
    for parametre in modele.parameters():
        parametre.requires_grad = False

    # On remplace la dernière couche par une nouvelle tête pour nos 2 classes.
    # Cette couche est neuve, donc entraînable (requires_grad=True par défaut).
    modele.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(2048, 2),   # 2048 = taille de sortie du corps du ResNet50
    )
    return modele


def mettre_en_mode_entrainement(modele):
    """Active le mode entraînement UNIQUEMENT sur la tête du réseau.

    Le corps gelé reste en mode évaluation : cela fige aussi les statistiques
    internes des couches BatchNorm, qui sinon continueraient de bouger à
    chaque image, même avec les poids gelés.
    """
    modele.eval()        # tout le réseau en mode évaluation...
    modele.fc.train()    # ...sauf la tête (pour que son Dropout fonctionne)


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
    print("=== Entraînement du ResNet50 (transfer learning) ===\n")

    # Graine fixée pour la reproductibilité (même graine que le CNN :
    # la comparaison se fait à conditions égales).
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
    modele = creer_resnet50().to(device)
    total = sum(p.numel() for p in modele.parameters())
    entrainables = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    print(f"Paramètres : {entrainables:,} entraînables sur {total:,} "
          f"({100 * entrainables / total:.2f}%)\n")

    loss_fn = nn.CrossEntropyLoss()
    # On ne donne à l'optimiseur QUE les paramètres de la tête.
    optimizer = torch.optim.AdamW(modele.fc.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ----- 3. Boucle d'entraînement (identique à celle du CNN) -----
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
    torch.save(best_state, DOSSIER_SORTIES / "resnet50.pth")
    with open(DOSSIER_SORTIES / "historique_resnet50.json", "w", encoding="utf-8") as f:
        json.dump(historique, f, indent=2)
    print("Modèle sauvegardé : outputs/resnet50.pth")

    # ----- 5. Courbes d'apprentissage -----
    (DOSSIER_SORTIES / "figures").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(historique["train_loss"], label="perte entraînement (train_loss)")
    plt.plot(historique["val_loss"], label="perte validation (val_loss)")
    plt.xlabel("Epoque")
    plt.ylabel("Perte (entropie croisée)")
    plt.title("Courbes d'apprentissage - ResNet50")
    plt.legend()
    plt.grid(True)
    plt.savefig(DOSSIER_SORTIES / "figures" / "courbes_resnet50.png")
    plt.close()
    print("Courbes sauvegardées : outputs/figures/courbes_resnet50.png")
    print("\nEtape suivante : python src/evaluation.py")
