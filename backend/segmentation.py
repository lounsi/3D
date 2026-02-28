"""
BrainXR - Module de Segmentation
=================================
Etape 2 du pipeline IA :
- Architecture U-Net simplifiee pour segmentation cerebrale
- Segmentation slice par slice
- Fallback : seuillage Otsu si pas de modele pre-entraine
"""

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  Architecture U-Net Simplifiee
# ============================================================

class DoubleConv(nn.Module):
    """
    Bloc de double convolution : (Conv2D -> BN -> ReLU) x 2
    Brique de base du U-Net.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net simplifie pour la segmentation d'images medicales.
    
    Architecture :
    - Encoder : 4 niveaux de downsampling
    - Bottleneck
    - Decoder : 4 niveaux de upsampling avec skip connections
    - Sortie : masque binaire (1 canal, sigmoide)
    
    Adapte pour images en niveaux de gris (1 canal d'entree).
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: list = None):
        super().__init__()
        
        if features is None:
            features = [32, 64, 128, 256]  # Plus leger que le U-Net original
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Encoder ---
        for feature in features:
            self.encoders.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # --- Decoder ---
        for feature in reversed(features):
            # Upsampling + concatenation avec skip connection
            self.decoders.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(feature * 2, feature))
        
        # --- Couche finale ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder (parcours inverse des skip connections)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Ajuster la taille si necessaire (padding)
            if x.shape != skip.shape:
                x = F.pad(x, [
                    0, skip.shape[3] - x.shape[3],
                    0, skip.shape[2] - x.shape[2]
                ])
            
            x = torch.cat((skip, x), dim=1)  # Concatenation
            x = self.decoders[idx + 1](x)    # Double conv
        
        return torch.sigmoid(self.final_conv(x))


# ============================================================
#  Chargement du modele
# ============================================================

def load_model(weights_path: str = None, device: str = "cpu") -> UNet:
    """
    Charge le modele U-Net, avec poids pre-entraines si disponibles.
    
    Args:
        weights_path: chemin vers les poids (.pth). None = modele vierge.
        device: 'cpu' ou 'cuda'
        
    Returns:
        Modele U-Net pret a l'inference
    """
    model = UNet(in_channels=1, out_channels=1)
    
    if weights_path is not None:
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"  -> Poids charges depuis '{weights_path}'")
        except Exception as e:
            print(f"  /!\ Impossible de charger les poids : {e}")
            print(f"  -> Utilisation du modele sans pre-entrainement")
    
    model.to(device)
    model.eval()
    return model


# ============================================================
#  Segmentation par U-Net
# ============================================================

def segment_slice_unet(model: UNet, slice_2d: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Segmente une seule slice 2D via le modele U-Net.
    
    Args:
        model: modele U-Net charge
        slice_2d: image 2D normalisee [0, 1], shape (H, W)
        device: 'cpu' ou 'cuda'
        
    Returns:
        Masque binaire (H, W) avec 0 = fond, 1 = cerveau
    """
    # Preparation du tensor : (1, 1, H, W)
    tensor = torch.FloatTensor(slice_2d).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(tensor)
    
    # Conversion en masque binaire (seuil 0.5)
    mask = prediction.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.float64)
    
    return mask


# ============================================================
#  Segmentation par seuillage Otsu (Fallback)
# ============================================================

def segment_slice_otsu(slice_2d: np.ndarray) -> np.ndarray:
    """
    Segmente une slice via seuillage Otsu + operations morphologiques.
    
    Methode robuste utilisee comme fallback quand le U-Net 
    n'a pas de poids pre-entraines.
    
    Args:
        slice_2d: image 2D normalisee [0, 1], shape (H, W)
        
    Returns:
        Masque binaire (H, W) avec 0 = fond, 1 = cerveau
    """
    # Conversion en uint8 pour OpenCV
    slice_uint8 = (slice_2d * 255).astype(np.uint8)
    
    # Seuillage Otsu
    _, mask = cv2.threshold(slice_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Operations morphologiques pour nettoyer le masque
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Fermeture (combler les petits trous)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Ouverture (supprimer le bruit)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Garder uniquement le plus grand composant connexe (le cerveau)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    if num_labels > 1:
        # Trouver le plus grand composant (ignorer le fond = label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    
    return (mask / 255.0).astype(np.float64)


# ============================================================
#  Pipeline de segmentation complet
# ============================================================

def segment_volume(
    volume: np.ndarray,
    model: UNet = None,
    weights_path: str = None,
    device: str = "cpu",
    method: str = "auto"
) -> np.ndarray:
    """
    Segmente l'ensemble du volume slice par slice.
    
    Strategie :
    - Si un modele U-Net avec poids est disponible -> utiliser U-Net
    - Sinon -> fallback sur seuillage Otsu
    
    Args:
        volume: volume 3D pretraite (slices, H, W) normalise [0, 1]
        model: modele U-Net (optionnel)
        weights_path: chemin vers les poids du modele (optionnel)
        device: 'cpu' ou 'cuda'
        method: 'unet', 'otsu', ou 'auto' (auto-detection)
        
    Returns:
        Volume de masques binaires (slices, H, W)
    """
    print("\n" + "=" * 50)
    print("ETAPE 2 - SEGMENTATION")
    print("=" * 50)
    
    num_slices = volume.shape[0]
    masks = np.zeros_like(volume)
    
    # --- Choix de la methode ---
    use_unet = False
    
    if method == "auto":
        if weights_path is not None or model is not None:
            use_unet = True
        else:
            use_unet = False
    elif method == "unet":
        use_unet = True
    elif method == "otsu":
        use_unet = False
    
    if use_unet:
        print("\n-> Methode : U-Net (deep learning)")
        if model is None:
            model = load_model(weights_path, device)
        
        for i in tqdm(range(num_slices), desc="Segmentation U-Net"):
            masks[i] = segment_slice_unet(model, volume[i], device)
    else:
        print("\n-> Methode : Seuillage Otsu + morphologie")
        print("  (Pas de modele U-Net disponible - utilisation du fallback)")
        
        for i in tqdm(range(num_slices), desc="Segmentation Otsu"):
            masks[i] = segment_slice_otsu(volume[i])
    
    # --- Statistiques ---
    total_voxels = masks.size
    brain_voxels = int(masks.sum())
    ratio = brain_voxels / total_voxels * 100
    
    print(f"\n[OK] Segmentation terminee")
    print(f"   Slices traitees : {num_slices}")
    print(f"   Voxels cerveau : {brain_voxels:,} / {total_voxels:,} ({ratio:.1f}%)")
    
    return masks


# ============================================================
#  Test standalone
# ============================================================

if __name__ == "__main__":
    print("Test de segmentation Otsu sur image synthetique...")
    
    # Creer une image de test : cercle blanc sur fond noir
    test_slice = np.zeros((256, 256), dtype=np.float64)
    cv2.circle(test_slice, (128, 128), 80, 1.0, -1)
    
    # Ajouter du bruit
    noise = np.random.normal(0, 0.1, test_slice.shape)
    test_slice = np.clip(test_slice + noise, 0, 1)
    
    # Segmenter
    mask = segment_slice_otsu(test_slice)
    
    print(f"  Image : {test_slice.shape}, range [{test_slice.min():.2f}, {test_slice.max():.2f}]")
    print(f"  Masque : {mask.shape}, pixels segmentes = {int(mask.sum())}")
    print("  [OK] Test OK")
