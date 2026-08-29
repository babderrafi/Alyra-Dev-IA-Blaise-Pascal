# ============================================================================
# CONSTRUCTION DE dataBis/ : PROVENANCES AFFINEES PAR ANALYSE D'IMAGE
# ============================================================================
# Constat (fait en regardant les images) : l'etiquette de provenance srcA,
# deduite du nom de fichier, regroupe en realite DEUX modalites d'imagerie :
#   - fond clair  : la particule apparait sombre sur fond lumineux
#   - champ sombre : la particule apparait brillante sur fond noir
#
# Ce script transforme ce constat visuel en mesure :
#   1. pour chaque image, on mesure la luminosite du FOND (les 4 coins,
#      la particule etant centree)
#   2. la distribution est bimodale, avec une vallee quasi vide vers 0,45 :
#      en dessous = champ sombre, au dessus = fond clair
#   3. on copie data/ vers dataBis/ en affinant le nom de la source :
#      fibre_srcA_0001.jpg -> fibre_srcA-clair_0001.jpg (ou srcA-sombre)
#
# data/ n'est PAS modifie. dataBis/ garde la meme structure split/classe :
# tous les scripts du projet fonctionnent dessus a l'identique, et
# l'evaluation par provenance devient plus fine.
#
# Lancement : python src/construire_databis.py

import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_DONNEES = DOSSIER_PROJET / "data"
DOSSIER_DONNEES_BIS = DOSSIER_PROJET / "dataBis"

SPLITS = ["train", "val", "test"]
CLASSES = ["fibre", "fragment"]

# Le seuil vient de l'histogramme des 550 fonds : la classe "sombre" s'arrete
# vers 0,40 et la classe "claire" commence vers 0,45 — entre les deux, une
# seule image. 0,45 coupe donc dans la vallee.
SEUIL_FOND_SOMBRE = 0.45

# On ne cree un sous-groupe (srcX-clair / srcX-sombre) que si la modalite
# minoritaire compte au moins 10 images : en dessous, ce serait un groupe
# trop petit pour etre analyse, on garde le nom d'origine.
MINIMUM_PAR_MODALITE = 10


def luminosite_fond(chemin, k=28):
    """Luminosite moyenne des 4 coins de l'image (la particule est centree,
    les coins ne contiennent donc que du fond)."""
    a = np.asarray(Image.open(chemin).convert("L").resize((224, 224)),
                   dtype=np.float32) / 255.0
    return float(np.mean([a[:k, :k].mean(), a[:k, -k:].mean(),
                          a[-k:, :k].mean(), a[-k:, -k:].mean()]))


print("=== Construction de dataBis/ (provenances affinees) ===\n")

# ----------------------------------------------------------------------------
# 1. Mesurer le fond de chaque image et compter les modalites par source
# ----------------------------------------------------------------------------
print("[1] Mesure du fond des 550 images...")
fiches = []   # (chemin, split, classe, source, modalite)
compte_modalites = defaultdict(lambda: {"clair": 0, "sombre": 0})
for split in SPLITS:
    for classe in CLASSES:
        for chemin in sorted((DOSSIER_DONNEES / split / classe).glob("*.jpg")):
            source = chemin.name.split("_")[1]
            modalite = "sombre" if luminosite_fond(chemin) < SEUIL_FOND_SOMBRE else "clair"
            fiches.append((chemin, split, classe, source, modalite))
            compte_modalites[source][modalite] += 1

print(f"    {'source':8s} {'fond clair':>11s} {'champ sombre':>13s} {'on separe ?':>12s}")
sources_separees = set()
for source in sorted(compte_modalites):
    c = compte_modalites[source]
    separer = min(c["clair"], c["sombre"]) >= MINIMUM_PAR_MODALITE
    if separer:
        sources_separees.add(source)
    print(f"    {source:8s} {c['clair']:11d} {c['sombre']:13d} "
          f"{'OUI' if separer else 'non':>12s}")

# ----------------------------------------------------------------------------
# 2. Copier vers dataBis/ avec les noms affines
# ----------------------------------------------------------------------------
print("\n[2] Copie vers dataBis/ ...")
if DOSSIER_DONNEES_BIS.exists():
    shutil.rmtree(DOSSIER_DONNEES_BIS)   # on repart de zero a chaque execution

for chemin, split, classe, source, modalite in fiches:
    # fibre_srcA_0001.jpg -> fibre_srcA-sombre_0001.jpg (si srcA est separee)
    if source in sources_separees:
        nouveau_nom = chemin.name.replace(f"_{source}_", f"_{source}-{modalite}_")
    else:
        nouveau_nom = chemin.name
    destination = DOSSIER_DONNEES_BIS / split / classe / nouveau_nom
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chemin, destination)
print(f"    {len(fiches)} images copiees (data/ n'a pas ete touche).")

# ----------------------------------------------------------------------------
# 3. L'audit qui justifie tout : composition par provenance affinee
# ----------------------------------------------------------------------------
print("\n[3] Composition par provenance AFFINEE (le tableau a montrer)")
composition = defaultdict(lambda: {"fibre": 0, "fragment": 0})
for chemin, split, classe, source, modalite in fiches:
    source_fine = f"{source}-{modalite}" if source in sources_separees else source
    composition[source_fine][classe] += 1

print(f"    {'source fine':14s} {'fibre':>6s} {'fragment':>9s} {'total':>6s} {'% fibres':>9s}")
for source_fine in sorted(composition):
    c = composition[source_fine]
    total = c["fibre"] + c["fragment"]
    part = 100 * c["fibre"] / total
    alerte = "  <- desequilibre : indice exploitable !" if abs(part - 50) > 10 else ""
    print(f"    {source_fine:14s} {c['fibre']:6d} {c['fragment']:9d} {total:6d} "
          f"{part:8.1f}%{alerte}")

print("\n    Lecture : les groupes desequilibres quantifient le biais residuel")
print("    de srcA — le fond y est correle a la classe. Ce biais etait")
print("    invisible au niveau 'srcA' (50.0% en apparence) : l'affinage le")
print("    rend mesurable. On ne peut PAS l'equilibrer (trop peu de fibres")
print("    en champ sombre) : on le documente, et l'evaluation par provenance")
print("    fine permet de juger le modele la ou le fond n'aide pas.")
print("\ndataBis/ est pret. Pour l'utiliser : pointer DOSSIER_DONNEES vers")
print("dataBis dans src/preparation.py (une ligne), tout le reste est identique.")
