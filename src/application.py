# ============================================================================
# APPLICATION DE DEMONSTRATION (déploiement local avec Streamlit)
# ============================================================================
# Une petite interface web : on téléverse une photo de particule et le
# ResNet50 entraîné prédit "fibre" ou "fragment" avec sa confiance.
#
# Lancement : streamlit run src/application.py
# (puis ouvrir http://localhost:8501 dans le navigateur)

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CHEMIN_POIDS = Path(__file__).resolve().parent.parent / "outputs" / "resnet50.pth"
CLASSES = ["fibre", "fragment"]   # ordre alphabétique, comme à l'entraînement

# Les MEMES transformations que pour la validation : l'image doit être
# préparée exactement comme pendant l'entraînement, sinon les prédictions
# seraient faussées.
transfo = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource   # le modèle n'est chargé qu'une fois, pas à chaque clic
def charger_modele():
    """Reconstruit le ResNet50 et charge les poids entraînés."""
    modele = models.resnet50(weights=None)   # architecture vide (nos poids arrivent)
    modele.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 2))
    modele.load_state_dict(torch.load(CHEMIN_POIDS, map_location="cpu"))
    modele.eval()   # mode évaluation : indispensable pour prédire
    return modele


st.title("Fibre ou fragment ?")
st.write("Classification de microplastiques par deep learning "
         "(ResNet50 + transfer learning) — projet Alyra.")

if not CHEMIN_POIDS.is_file():
    st.error("Modèle introuvable. Lancer d'abord : python src/entrainement_resnet50.py")
    st.stop()

fichier = st.file_uploader("Choisir une image de particule",
                           type=["jpg", "jpeg", "png"])

if fichier is not None:
    image = Image.open(fichier).convert("RGB")
    st.image(image, caption="Image chargée", width=300)

    # unsqueeze(0) ajoute la dimension "lot" : le modèle attend un lot
    # d'images, ici un lot d'une seule image.
    entree = transfo(image).unsqueeze(0)
    with torch.no_grad():
        sorties = charger_modele()(entree)
        # softmax transforme les scores bruts en probabilités (somme = 1)
        probabilites = torch.softmax(sorties, dim=1)[0]

    indice = int(probabilites.argmax())
    st.subheader(f"Prédiction : {CLASSES[indice]}")
    st.write(f"Confiance : {probabilites[indice]:.1%}")
    st.progress(float(probabilites[indice]))
    st.caption(f"Détail — fibre : {probabilites[0]:.1%}, "
               f"fragment : {probabilites[1]:.1%}")
