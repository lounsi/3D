"""
BrainXR - Module de Pretraitement
==================================
Etape 1 du pipeline IA :
- Chargement des images IRM 2D (PNG, JPG, NIfTI)
- Normalisation des intensites
- Redimensionnement uniforme
- Reduction du bruit
- Conversion en volume 3D numpy
"""

import os
import glob
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# --- Tentative d'import optionnel pour NIfTI ---
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False


# ============================================================
#  Chargement des images
# ============================================================

def load_slices_from_images(input_dir: str) -> np.ndarray:
    """
    Charge une serie d'images 2D (PNG/JPG) depuis un dossier 
    et les empile en volume 3D.
    
    Args:
        input_dir: chemin vers le dossier contenant les images
        
    Returns:
        Volume 3D numpy (slices, H, W) en float64 [0-255]
    """
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # Tri naturel par nom de fichier
    files.sort()
    
    if len(files) == 0:
        raise FileNotFoundError(
            f"Aucune image trouvee dans '{input_dir}'. "
            f"Formats supportes : PNG, JPG, BMP, TIFF"
        )
    
    slices = []
    for f in tqdm(files, desc="Chargement des images"):
        img = Image.open(f).convert('L')  # Conversion en niveaux de gris
        slices.append(np.array(img, dtype=np.float64))
    
    print(f"  -&gt; {len(slices)} slices chargees depuis '{input_dir}'")
    return np.stack(slices, axis=0)


def load_nifti(filepath: str) -> np.ndarray:
    """
    Charge un fichier NIfTI (.nii ou .nii.gz) en volume 3D.
    
    Args:
        filepath: chemin vers le fichier NIfTI
        
    Returns:
        Volume 3D numpy (slices, H, W) en float64
    """
    if not HAS_NIBABEL:
        raise ImportError(
            "nibabel n'est pas installe. "
            "Installez-le avec : pip install nibabel"
        )
    
    nii = nib.load(filepath)
    volume = nii.get_fdata().astype(np.float64)
    
    # S'assurer que le volume est 3D
    if volume.ndim == 4:
        volume = volume[:, :, :, 0]
    
    # Transposer pour avoir (slices, H, W) - axial view
    volume = np.transpose(volume, (2, 0, 1))
    
    print(f"  -&gt; Volume NIfTI charge : {volume.shape}")
    return volume


def load_dicom_series(input_dir: str) -> np.ndarray:
    """
    Charge une serie DICOM depuis un dossier via SimpleITK.
    
    Args:
        input_dir: chemin vers le dossier contenant les fichiers DICOM
        
    Returns:
        Volume 3D numpy (slices, H, W) en float64
    """
    if not HAS_SIMPLEITK:
        raise ImportError(
            "SimpleITK n'est pas installe. "
            "Installez-le avec : pip install SimpleITK"
        )
    
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(input_dir)
    
    if len(dicom_files) == 0:
        raise FileNotFoundError(
            f"Aucun fichier DICOM trouve dans '{input_dir}'"
        )
    
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image).astype(np.float64)
    
    print(f"  -&gt; Serie DICOM chargee : {volume.shape}")
    return volume


# ============================================================
#  Pretraitement
# ============================================================

def normalize(volume: np.ndarray) -> np.ndarray:
    """
    Normalise les intensites du volume dans l'intervalle [0, 1].
    
    Args:
        volume: volume 3D numpy
        
    Returns:
        Volume normalise [0, 1]
    """
    v_min = volume.min()
    v_max = volume.max()
    
    if v_max - v_min < 1e-8:
        print("  /!\ Volume constant detecte, normalisation ignoree")
        return np.zeros_like(volume)
    
    normalized = (volume - v_min) / (v_max - v_min)
    print(f"  -&gt; Normalisation : [{v_min:.1f}, {v_max:.1f}] -&gt; [0, 1]")
    return normalized


def resize_slices(volume: np.ndarray, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Redimensionne chaque slice du volume a une taille uniforme.
    
    Args:
        volume: volume 3D (slices, H, W)
        target_size: (height, width) cible
        
    Returns:
        Volume redimensionne
    """
    resized = np.zeros((volume.shape[0], target_size[0], target_size[1]))
    
    for i in range(volume.shape[0]):
        resized[i] = cv2.resize(
            volume[i], 
            (target_size[1], target_size[0]),  # cv2 utilise (width, height)
            interpolation=cv2.INTER_LINEAR
        )
    
    print(f"  -&gt; Redimensionnement : {volume.shape[1:]} -&gt; {target_size}")
    return resized


def denoise(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Applique un filtre gaussien pour reduire le bruit.
    
    Args:
        volume: volume 3D normalise
        sigma: ecart-type du filtre gaussien
        
    Returns:
        Volume debruite
    """
    denoised = gaussian_filter(volume, sigma=sigma)
    print(f"  -&gt; Debruitage gaussien (sigma={sigma})")
    return denoised


def apply_clahe_per_slice(volume: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Applique CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    sur chaque slice pour ameliorer le contraste.
    
    Args:
        volume: volume 3D normalise [0, 1]
        clip_limit: limite de clipping pour CLAHE
        
    Returns:
        Volume avec contraste ameliore
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = np.zeros_like(volume)
    
    for i in range(volume.shape[0]):
        slice_uint8 = (volume[i] * 255).astype(np.uint8)
        enhanced_slice = clahe.apply(slice_uint8)
        enhanced[i] = enhanced_slice.astype(np.float64) / 255.0
    
    print(f"  -&gt; Amelioration contraste CLAHE (clip={clip_limit})")
    return enhanced


# ============================================================
#  Pipeline complet
# ============================================================

def preprocess_pipeline(
    input_path: str,
    target_size: tuple = (256, 256),
    sigma: float = 1.0,
    use_clahe: bool = True
) -> np.ndarray:
    """
    Pipeline de pretraitement complet.
    
    Etapes :
    1. Chargement (auto-detection du format)
    2. Normalisation [0, 1]
    3. Redimensionnement
    4. Debruitage
    5. Amelioration contraste (optionnel)
    
    Args:
        input_path: chemin vers dossier d'images ou fichier NIfTI
        target_size: taille cible (H, W)
        sigma: sigma pour le filtre gaussien
        use_clahe: appliquer CLAHE pour ameliorer le contraste
        
    Returns:
        Volume 3D pretraite (slices, H, W) normalise [0, 1]
    """
    print("=" * 50)
    print("ETAPE 1 - PRETRAITEMENT")
    print("=" * 50)
    
    # --- 1. Chargement (auto-detection format) ---
    print("\n1.1 Chargement des donnees...")
    
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.nii', '.gz']:
            volume = load_nifti(input_path)
        else:
            raise ValueError(f"Format fichier non supporte : {ext}")
    elif os.path.isdir(input_path):
        # Verifier s'il y a des fichiers DICOM
        dcm_files = glob.glob(os.path.join(input_path, '*.dcm'))
        if dcm_files:
            volume = load_dicom_series(input_path)
        else:
            volume = load_slices_from_images(input_path)
    else:
        raise FileNotFoundError(f"Chemin introuvable : '{input_path}'")
    
    print(f"  -&gt; Volume brut : shape={volume.shape}, "
          f"dtype={volume.dtype}, "
          f"range=[{volume.min():.1f}, {volume.max():.1f}]")
    
    # --- 2. Normalisation ---
    print("\n1.2 Normalisation...")
    volume = normalize(volume)
    
    # --- 3. Redimensionnement ---
    print("\n1.3 Redimensionnement...")
    volume = resize_slices(volume, target_size)
    
    # --- 4. Debruitage ---
    print("\n1.4 Debruitage...")
    volume = denoise(volume, sigma)
    
    # --- 5. Amelioration contraste ---
    if use_clahe:
        print("\n1.5 Amelioration contraste...")
        volume = apply_clahe_per_slice(volume)
    
    print(f"\n[OK] Pretraitement termine : {volume.shape}")
    print(f"   Range finale : [{volume.min():.4f}, {volume.max():.4f}]")
    
    return volume


# ============================================================
#  Test standalone
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <input_dir_or_file>")
        print("  Supporte : dossier d'images PNG/JPG, fichier NIfTI, dossier DICOM")
        sys.exit(1)
    
    input_path = sys.argv[1]
    volume = preprocess_pipeline(input_path)
    print(f"\nResultat : volume shape = {volume.shape}")
