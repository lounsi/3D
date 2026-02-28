"""
BrainXR - Classificateur Alzheimer
====================================
Classification de cerveaux IRM en 3 categories :
- Sain (CDR = 0)
- Declin leger (CDR = 0.5)
- Alzheimer (CDR >= 1)

Approche : extraction de features volumetriques par atlas
+ reseau de neurones PyTorch (fully-connected).

Usage :
    python alzheimer_classifier.py
"""

import os
import sys
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
#  Extraction de features volumetriques
# ============================================================

def extract_volumetric_features(gray_matter_maps, n_subjects=None):
    """
    Extrait des features volumetriques a partir des cartes
    de matiere grise en utilisant l'atlas Harvard-Oxford.

    Pour chaque sujet :
    - Volume total de matiere grise
    - Volume par region de l'atlas
    - Ratios inter-regions

    Args:
        gray_matter_maps: liste de chemins vers les fichiers NIfTI
        n_subjects: nombre de sujets a traiter (None = tous)

    Returns:
        features: np.ndarray (n_subjects, n_features)
        region_names: liste des noms de regions
    """
    from nilearn import datasets, image
    import nibabel as nib

    print("\n-> Chargement de l'atlas Harvard-Oxford...")
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = image.load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    labels = atlas.labels  # noms des regions

    # Regions valides (exclure le fond = index 0)
    region_indices = list(range(1, len(labels)))
    region_names = [labels[i] for i in region_indices]

    if n_subjects is not None:
        gray_matter_maps = gray_matter_maps[:n_subjects]

    n_sub = len(gray_matter_maps)
    n_regions = len(region_indices)

    # Features : volume par region + volume total + ratios
    # Total features = n_regions + 1 (total) + n_regions (ratios vs total)
    n_features = n_regions * 2 + 1
    features = np.zeros((n_sub, n_features))

    print(f"-> Extraction des features pour {n_sub} sujets, {n_regions} regions...")

    for i, gm_path in enumerate(gray_matter_maps):
        # Charger et resampler sur l'atlas
        gm_img = image.load_img(gm_path)
        gm_resampled = image.resample_to_img(gm_img, atlas_img, interpolation='nearest')
        gm_data = gm_resampled.get_fdata()

        # Volume total de matiere grise
        total_gm = float(np.sum(gm_data))
        features[i, 0] = total_gm

        # Volume par region
        for j, ridx in enumerate(region_indices):
            mask = atlas_data == ridx
            region_vol = float(np.sum(gm_data[mask]))
            features[i, 1 + j] = region_vol

            # Ratio vs total
            if total_gm > 0:
                features[i, 1 + n_regions + j] = region_vol / total_gm
            else:
                features[i, 1 + n_regions + j] = 0.0

        if (i + 1) % 5 == 0 or i == n_sub - 1:
            print(f"   Sujet {i + 1}/{n_sub} traite")

    return features, region_names


def extract_single_subject_features(nifti_path, atlas_img=None, atlas_data=None, region_indices=None):
    """
    Extrait les features volumetriques pour un seul sujet.

    Args:
        nifti_path: chemin vers le fichier NIfTI du sujet
        atlas_img, atlas_data, region_indices: atlas pre-charge (optionnel)

    Returns:
        features: np.ndarray (1, n_features)
    """
    from nilearn import datasets, image
    import nibabel as nib

    if atlas_img is None:
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = image.load_img(atlas.maps)
        atlas_data = atlas_img.get_fdata()
        labels = atlas.labels
        region_indices = list(range(1, len(labels)))

    n_regions = len(region_indices)
    n_features = n_regions * 2 + 1
    features = np.zeros((1, n_features))

    gm_img = image.load_img(nifti_path)
    gm_resampled = image.resample_to_img(gm_img, atlas_img, interpolation='nearest')
    gm_data = gm_resampled.get_fdata()

    total_gm = float(np.sum(gm_data))
    features[0, 0] = total_gm

    for j, ridx in enumerate(region_indices):
        mask = atlas_data == ridx
        region_vol = float(np.sum(gm_data[mask]))
        features[0, 1 + j] = region_vol
        if total_gm > 0:
            features[0, 1 + n_regions + j] = region_vol / total_gm
        else:
            features[0, 1 + n_regions + j] = 0.0

    return features


# ============================================================
#  Labels CDR -> classes
# ============================================================

def cdr_to_label(cdr_value):
    """
    Convertit le score CDR en label de classe.
    CDR 0    -> 0 (Sain)
    CDR 0.5  -> 1 (Declin leger)
    CDR >= 1 -> 2 (Alzheimer)
    """
    if cdr_value == 0:
        return 0
    elif cdr_value == 0.5:
        return 1
    else:
        return 2


LABEL_NAMES = ["Sain", "Declin leger", "Alzheimer"]


# ============================================================
#  Modele PyTorch - Reseau fully-connected
# ============================================================

class AlzheimerClassifier(nn.Module):
    """
    Reseau de neurones simple (MLP) pour la classification
    Alzheimer a partir de features volumetriques.

    Architecture :
    - Input -> 128 -> ReLU -> Dropout
    - 128 -> 64 -> ReLU -> Dropout
    - 64 -> 3 (Softmax)
    """

    def __init__(self, n_features, n_classes=3, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
#  Entrainement
# ============================================================

def train_classifier(features, labels, n_epochs=200, lr=0.001, device="cpu"):
    """
    Entraine le classificateur sur les features volumetriques.

    Args:
        features: np.ndarray (n_subjects, n_features)
        labels: np.ndarray (n_subjects,) avec classes 0, 1, 2
        n_epochs: nombre d'epoques
        lr: learning rate
        device: 'cpu' ou 'cuda'

    Returns:
        model: modele entraine
        scaler_mean, scaler_std: parametres de normalisation
    """
    print(f"\n-> Entrainement du classificateur ({n_epochs} epoques)...")

    # Normalisation des features (z-score)
    scaler_mean = features.mean(axis=0)
    scaler_std = features.std(axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0
    features_norm = (features - scaler_mean) / scaler_std

    # Conversion en tenseurs
    X = torch.FloatTensor(features_norm).to(device)
    y = torch.LongTensor(labels).to(device)

    # Dataset et DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=min(16, len(X)), shuffle=True)

    # Modele
    n_features = features.shape[1]
    model = AlzheimerClassifier(n_features, n_classes=3).to(device)

    # Poids de classes (gerer le desequilibre)
    class_counts = np.bincount(labels, minlength=3).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 3
    weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Entrainement
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            acc = correct / total * 100
            print(f"   Epoch {epoch + 1}/{n_epochs} | Loss: {total_loss:.4f} | Acc: {acc:.1f}%")

    model.eval()
    return model, scaler_mean, scaler_std


# ============================================================
#  Inference
# ============================================================

def classify_brain(nifti_path, model=None, scaler_mean=None, scaler_std=None,
                   model_path=None, device="cpu"):
    """
    Classifie un cerveau IRM.

    Args:
        nifti_path: chemin vers le fichier NIfTI
        model: modele PyTorch (optionnel si model_path est fourni)
        scaler_mean, scaler_std: parametres de normalisation
        model_path: chemin vers les poids sauvegardes (.pth)
        device: 'cpu' ou 'cuda'

    Returns:
        dict avec prediction, confiance, probabilites
    """
    print(f"\n-> Classification du cerveau : {os.path.basename(nifti_path)}")

    # Charger le modele si necessaire
    if model is None and model_path is not None:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        n_features = checkpoint['n_features']
        model = AlzheimerClassifier(n_features).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler_mean = checkpoint['scaler_mean']
        scaler_std = checkpoint['scaler_std']
        model.eval()

    if model is None:
        raise ValueError("Aucun modele fourni. Fournir model ou model_path.")

    # Extraire les features
    features = extract_single_subject_features(nifti_path)

    # Normaliser
    features_norm = (features - scaler_mean) / scaler_std

    # Prediction
    X = torch.FloatTensor(features_norm).to(device)
    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class]) * 100

    result = {
        "prediction": LABEL_NAMES[predicted_class],
        "confidence": round(confidence, 1),
        "probabilities": {
            LABEL_NAMES[i]: round(float(probabilities[i]) * 100, 1)
            for i in range(3)
        }
    }

    print(f"   Prediction  : {result['prediction']}")
    print(f"   Confiance   : {result['confidence']}%")
    print(f"   Probabilites: {result['probabilities']}")

    return result


# ============================================================
#  Sauvegarde / Chargement du modele
# ============================================================

def save_model(model, scaler_mean, scaler_std, filepath):
    """Sauvegarde le modele et les parametres de normalisation."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler_mean,
        'scaler_std': scaler_std,
        'n_features': scaler_mean.shape[0]
    }
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"-> Modele sauvegarde : {filepath}")


# ============================================================
#  Pipeline complet : chargement OASIS + entrainement + test
# ============================================================

def full_pipeline(n_subjects=30, device="cpu"):
    """
    Pipeline complet :
    1. Telecharger OASIS
    2. Extraire les features
    3. Entrainer le classificateur
    4. Evaluer

    Returns:
        model, scaler_mean, scaler_std, oasis (dataset)
    """
    from nilearn import datasets

    print("=" * 50)
    print("  CLASSIFICATEUR ALZHEIMER")
    print("=" * 50)

    # 1. Telecharger OASIS
    print("\n-> Telechargement OASIS VBM...")
    oasis = datasets.fetch_oasis_vbm(n_subjects=n_subjects)

    # 2. Extraire les labels CDR (ext_vars est un DataFrame pandas)
    cdr_series = oasis.ext_vars['cdr']
    cdr_values = cdr_series.values
    
    # Gerer les NaN dans CDR
    valid_mask = ~np.isnan(cdr_values)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < len(cdr_values):
        print(f"   {len(cdr_values) - len(valid_indices)} sujets sans CDR (exclus)")

    labels = np.array([cdr_to_label(cdr_values[i]) for i in valid_indices])
    gray_matter_maps = [oasis.gray_matter_maps[i] for i in valid_indices]

    # Distribution des classes
    for i, name in enumerate(LABEL_NAMES):
        count = np.sum(labels == i)
        print(f"   {name}: {count} sujets")

    # 3. Extraire les features
    features, region_names = extract_volumetric_features(gray_matter_maps)

    # 4. Entrainer
    model, scaler_mean, scaler_std = train_classifier(
        features, labels, n_epochs=200, device=device
    )

    # 5. Evaluer sur tout le dataset (pas de split vu le peu de donnees)
    features_norm = (features - scaler_mean) / scaler_std
    X = torch.FloatTensor(features_norm).to(device)
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()

    accuracy = np.mean(predicted == labels) * 100
    print(f"\n-> Accuracy globale : {accuracy:.1f}%")

    # Matrice de confusion
    print("\n-> Matrice de confusion :")
    print(f"{'':>15} | {'Pred Sain':>10} | {'Pred Leger':>10} | {'Pred Alz':>10}")
    print("-" * 55)
    for i, name in enumerate(LABEL_NAMES):
        row = [np.sum((labels == i) & (predicted == j)) for j in range(3)]
        print(f"{name:>15} | {row[0]:>10} | {row[1]:>10} | {row[2]:>10}")

    # 6. Sauvegarder le modele
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "data", "models", "alzheimer_classifier.pth")
    save_model(model, scaler_mean, scaler_std, model_path)

    return model, scaler_mean, scaler_std, oasis


# ============================================================
#  Test standalone
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    model, scaler_mean, scaler_std, oasis = full_pipeline(
        n_subjects=30, device=device
    )

    # Tester sur un sujet aleatoire
    import random
    idx = random.randint(0, len(oasis.gray_matter_maps) - 1)
    test_path = oasis.gray_matter_maps[idx]

    print(f"\n{'=' * 50}")
    print(f"  TEST sur sujet #{idx + 1}")
    print(f"{'=' * 50}")

    result = classify_brain(
        test_path, model=model,
        scaler_mean=scaler_mean, scaler_std=scaler_std,
        device=device
    )

    # Afficher les infos demographiques
    cdr = oasis.ext_vars['cdr'].iloc[idx]
    print(f"\n   CDR reel : {cdr} ({LABEL_NAMES[cdr_to_label(cdr)]})")
    print(f"   Prediction : {result['prediction']} ({result['confidence']}%)")
