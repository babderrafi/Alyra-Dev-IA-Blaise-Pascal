# ============================================================================
# ENTRAINEMENT DU CNN SIMPLE (entraîné de zéro, sans pré-entraînement)
# ============================================================================
# Ce petit réseau sert de point de comparaison au ResNet50 : il permet de
# répondre à la question "qu'apporte vraiment le transfer learning ?".
# Comme il part de zéro avec seulement 384 images, on s'attend à ce qu'il
# soit moins bon et plus lent à converger.
#
# Lancement : python src/entrainement_cnn.py

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

from preparation import charger_donnees, SEED

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_SORTIES = DOSSIER_PROJET / "outputs"

# ----- Hyperparamètres -----
EPOCHS = 80          # plafond généreux : parti de zéro, le CNN converge lentement
PATIENCE = 10        # arrêt anticipé si la validation stagne 10 époques de suite
LR = 1e-3            # learning rate : la taille des pas de correction
WEIGHT_DECAY = 1e-4  # régularisation L2, contre le sur-apprentissage


def creer_cnn():
    """Construit un petit réseau convolutif : 3 blocs de convolution.

    Chaque bloc : convolution -> BatchNorm -> ReLU -> MaxPool.
    A chaque bloc, l'image est deux fois plus petite et on double le nombre
    de filtres. Environ 94 000 paramètres au total (contre 23,5 millions
    pour le ResNet50).
    """
    return nn.Sequential(
        # Bloc 1 : 3 canaux -> 32 filtres
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Bloc 2 : 32 -> 64 filtres
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Bloc 3 : 64 -> 128 filtres
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # On moyenne chaque carte de features en un seul chiffre,
        # puis une couche linéaire donne les 2 scores de classes.
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),      # éteint 30% des neurones au hasard (anti sur-apprentissage)
        nn.Linear(128, 2),
    )


def evaluer(modele, chargeur, loss_fn, device):
    """Calcule la perte moyenne et l'accuracy du modèle sur un jeu de données."""
    modele.eval()                 # mode évaluation : désactive dropout et fige BatchNorm
    total_loss = 0.0
    bonnes_reponses = 0
    with torch.no_grad():         # pas de calcul de gradients : plus rapide
        for images, etiquettes in chargeur:
            images, etiquettes = images.to(device), etiquettes.to(device)
            sorties = modele(images)
            total_loss += loss_fn(sorties, etiquettes).item() * len(images)
            predictions = sorties.argmax(dim=1)
            bonnes_reponses += (predictions == etiquettes).sum().item()
    n = len(chargeur.dataset)
    return total_loss / n, bonnes_reponses / n


if __name__ == "__main__":
    print("=== Entraînement du CNN simple (de zéro) ===\n")

    # Graine fixée : sans elle, deux entraînements donnent des résultats
    # différents et la comparaison avec le ResNet50 ne veut plus rien dire.
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
    modele = creer_cnn().to(device)
    nb_parametres = sum(p.numel() for p in modele.parameters())
    print(f"Paramètres du modèle : {nb_parametres:,} (tous entraînés)\n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(modele.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    # Le learning rate diminue progressivement au fil des époques :
    # grands pas au début pour apprendre vite, petits pas à la fin pour affiner.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ----- 3. Boucle d'entraînement -----
    historique = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_sans_progres = 0
    debut = time.time()

    for epoch in range(EPOCHS):
        # -- Phase d'entraînement : le modèle apprend --
        modele.train()
        loss_cumulee = 0.0
        for images, etiquettes in chargeur_train:
            images, etiquettes = images.to(device), etiquettes.to(device)
            optimizer.zero_grad()                       # remise à zéro des gradients
            loss = loss_fn(modele(images), etiquettes)
            loss.backward()                             # calcul des gradients
            optimizer.step()                            # mise à jour des poids
            loss_cumulee += loss.item() * len(images)
        train_loss = loss_cumulee / len(chargeur_train.dataset)
        scheduler.step()

        # -- Phase de validation : on mesure sans apprendre --
        val_loss, val_acc = evaluer(modele, chargeur_val, loss_fn, device)

        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)
        historique["val_acc"].append(val_acc)
        print(f"Epoque {epoch + 1:3d}/{EPOCHS} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

        # -- On garde le meilleur modèle vu jusqu'ici (selon la perte val) --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(modele.state_dict())
            epochs_sans_progres = 0
        else:
            epochs_sans_progres += 1
            # Arrêt anticipé : si la validation ne s'améliore plus,
            # continuer ne ferait que sur-apprendre.
            if epochs_sans_progres >= PATIENCE:
                print(f"\nArrêt anticipé : pas d'amélioration depuis {PATIENCE} époques.")
                break

    duree = time.time() - debut
    print(f"\nEntraînement terminé en {duree / 60:.1f} min "
          f"({epoch + 1} époques). Meilleure val_loss : {best_val_loss:.4f}")

    # ----- 4. Sauvegarde du meilleur modèle et de l'historique -----
    DOSSIER_SORTIES.mkdir(exist_ok=True)
    torch.save(best_state, DOSSIER_SORTIES / "cnn.pth")
    with open(DOSSIER_SORTIES / "historique_cnn.json", "w", encoding="utf-8") as f:
        json.dump(historique, f, indent=2)
    print("Modèle sauvegardé : outputs/cnn.pth")

    # ----- 5. Courbes d'apprentissage -----
    # L'écart entre les deux courbes est le diagnostic du sur-apprentissage :
    # si la perte train baisse mais que la perte val remonte, le modèle
    # apprend le jeu d'entraînement par coeur.
    (DOSSIER_SORTIES / "figures").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(historique["train_loss"], label="perte entraînement (train_loss)")
    plt.plot(historique["val_loss"], label="perte validation (val_loss)")
    plt.xlabel("Epoque")
    plt.ylabel("Perte (entropie croisée)")
    plt.title("Courbes d'apprentissage - CNN simple")
    plt.legend()
    plt.grid(True)
    plt.savefig(DOSSIER_SORTIES / "figures" / "courbes_cnn.png")
    plt.close()
    print("Courbes sauvegardées : outputs/figures/courbes_cnn.png")
    print("\nEtape suivante : python src/entrainement_resnet50.py")
