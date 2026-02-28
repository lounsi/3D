"""
BrainXR - Module de Reconstruction 3D
=======================================
Etape 3 du pipeline IA :
- Empilement des masques 2D en volume 3D
- Lissage du volume
- Extraction de surface via Marching Cubes
- Simplification de maillage (optionnel)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.measure import marching_cubes
from tqdm import tqdm

# --- Import optionnel pour simplification ---
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


# ============================================================
#  Construction du volume 3D
# ============================================================

def build_volume(masks: np.ndarray) -> np.ndarray:
    """
    Construit un volume binaire 3D a partir des masques 2D empiles.
    
    Applique un remplissage des trous dans chaque slice
    pour assurer la coherence volumetrique.
    
    Args:
        masks: masques binaires (slices, H, W)
        
    Returns:
        Volume binaire 3D nettoye
    """
    print("\n3.1 Construction du volume 3D...")
    
    volume = np.zeros_like(masks, dtype=np.float64)
    
    for i in range(masks.shape[0]):
        # Remplir les trous dans chaque slice
        binary_slice = masks[i] > 0.5
        filled = binary_fill_holes(binary_slice)
        volume[i] = filled.astype(np.float64)
    
    # Ajouter du padding vide sur TOUS les axes (Z, Y, X)
    # Cela evite que Marching Cubes cree des "couvercles" plats
    # quand le cerveau touche le bord du volume
    pad_size = 5
    volume = np.pad(volume, pad_width=pad_size, mode='constant', constant_values=0)
    
    non_empty = np.sum(np.any(volume > 0, axis=(1, 2)))
    print(f"  -> Volume construit : {volume.shape}")
    print(f"  -> Slices non-vides : {non_empty} / {volume.shape[0]}")
    print(f"  -> Padding 3D ajoute : {pad_size} voxels sur chaque bord")
    
    return volume


# ============================================================
#  Lissage du volume
# ============================================================

def smooth_volume(volume: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Applique un lissage gaussien 3D pour obtenir une surface plus lisse.
    
    Le lissage est crucial pour que Marching Cubes genere un mesh
    de qualite sans artefacts en escalier.
    
    Args:
        volume: volume binaire 3D
        sigma: ecart-type du filtre gaussien 3D
        
    Returns:
        Volume lisse (valeurs continues entre 0 et 1)
    """
    print(f"\n3.2 Lissage gaussien 3D (sigma={sigma})...")
    smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma)
    
    print(f"  -> Range apres lissage : [{smoothed.min():.4f}, {smoothed.max():.4f}]")
    return smoothed


# ============================================================
#  Marching Cubes - Extraction de surface
# ============================================================

def marching_cubes_reconstruction(
    volume: np.ndarray,
    threshold: float = 0.5,
    step_size: int = 1,
    spacing: tuple = (1.0, 1.0, 1.0)
) -> tuple:
    """
    Extrait la surface 3D du volume via l'algorithme Marching Cubes.
    
    Marching Cubes parcourt le volume voxel par voxel et genere
    des triangles a la frontiere entre les regions segmentees
    et le fond.
    
    Args:
        volume: volume 3D lisse (valeurs continues)
        threshold: seuil isosurface (typiquement 0.5)
        step_size: pas d'echantillonnage (1 = pleine resolution)
        spacing: espacement entre voxels (z, y, x)
        
    Returns:
        tuple (vertices, faces, normals) :
        - vertices : coordonnees 3D des sommets (N, 3)
        - faces : indices des triangles (M, 3)
        - normals : normales aux sommets (N, 3)
    """
    print(f"\n3.3 Marching Cubes (seuil={threshold}, pas={step_size})...")
    
    # Verifier que le volume contient des donnees au-dessus du seuil
    if volume.max() < threshold:
        raise ValueError(
            f"Le volume ne contient pas de valeurs au-dessus du seuil ({threshold}). "
            f"Max = {volume.max():.4f}. Essayez un seuil plus bas."
        )
    
    # Extraction via scikit-image
    vertices, faces, normals, _ = marching_cubes(
        volume,
        level=threshold,
        step_size=step_size,
        spacing=spacing
    )
    
    print(f"  -> Vertices : {vertices.shape[0]:,}")
    print(f"  -> Faces    : {faces.shape[0]:,}")
    
    return vertices, faces, normals


# ============================================================
#  Simplification du maillage
# ============================================================

def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_ratio: float = 0.5
) -> tuple:
    """
    Simplifie le maillage en reduisant le nombre de faces.
    
    Utilise trimesh pour la decimation. Utile pour reduire la 
    complexite du modele pour l'affichage en temps reel (XR).
    
    Args:
        vertices: sommets (N, 3)
        faces: faces triangulaires (M, 3)
        target_ratio: ratio de faces a conserver (0.5 = garder 50%)
        
    Returns:
        tuple (vertices_simplified, faces_simplified)
    """
    if not HAS_TRIMESH:
        print("  /!\ trimesh non installe, simplification ignoree")
        return vertices, faces
    
    print(f"\n3.4 Simplification du maillage (ratio={target_ratio})...")
    
    original_faces = faces.shape[0]
    target_faces = int(original_faces * target_ratio)
    
    # Creer un mesh trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Simplification
    simplified = mesh.simplify_quadric_decimation(target_faces)
    
    new_vertices = np.array(simplified.vertices)
    new_faces = np.array(simplified.faces)
    
    print(f"  -> Faces : {original_faces:,} -> {new_faces.shape[0]:,} "
          f"({new_faces.shape[0] / original_faces * 100:.1f}% conservees)")
    
    return new_vertices, new_faces


# ============================================================
#  Centrage et mise a l'echelle
# ============================================================

def center_and_scale(vertices: np.ndarray, target_size: float = 1.0) -> np.ndarray:
    """
    Centre le modele a l'origine et le met a l'echelle.
    
    Important pour l'import dans Unity : le modele doit etre
    centre et a une taille raisonnable.
    
    Args:
        vertices: sommets (N, 3)
        target_size: taille maximale souhaitee
        
    Returns:
        Vertices centres et mis a l'echelle
    """
    # Centrage a l'origine
    centroid = vertices.mean(axis=0)
    vertices_centered = vertices - centroid
    
    # Mise a l'echelle
    max_extent = np.max(np.abs(vertices_centered))
    if max_extent > 0:
        scale = target_size / max_extent
        vertices_centered *= scale
    
    print(f"  -> Modele centre et mis a l'echelle (taille max = {target_size})")
    
    return vertices_centered


# ============================================================
#  Pipeline de reconstruction complet
# ============================================================

def reconstruction_pipeline(
    masks: np.ndarray,
    sigma: float = 1.5,
    threshold: float = 0.5,
    step_size: int = 1,
    simplify: bool = True,
    simplify_ratio: float = 0.5,
    center: bool = True,
    target_size: float = 1.0,
    spacing: tuple = (1.0, 1.0, 1.0)
) -> tuple:
    """
    Pipeline complet de reconstruction 3D.
    
    Etapes :
    1. Construction du volume 3D
    2. Lissage gaussien
    3. Marching Cubes
    4. Simplification (optionnel)
    5. Centrage et mise a l'echelle
    
    Args:
        masks: masques de segmentation (slices, H, W)
        sigma: sigma pour lissage
        threshold: seuil pour Marching Cubes
        step_size: pas d'echantillonnage
        simplify: activer la simplification
        simplify_ratio: ratio de decimation
        center: centrer le modele
        target_size: taille maximale du modele
        spacing: espacement inter-voxels (z, y, x)
        
    Returns:
        tuple (vertices, faces, normals)
    """
    print("\n" + "=" * 50)
    print("ETAPE 3 - RECONSTRUCTION 3D")
    print("=" * 50)
    
    # 1. Construire le volume
    volume = build_volume(masks)
    
    # 2. Lisser
    smoothed = smooth_volume(volume, sigma)
    
    # 3. Marching Cubes
    vertices, faces, normals = marching_cubes_reconstruction(
        smoothed, threshold, step_size, spacing
    )
    
    # 4. Simplification
    if simplify and HAS_TRIMESH:
        vertices, faces = simplify_mesh(vertices, faces, simplify_ratio)
        # Recalculer les normales apres simplification
        if HAS_TRIMESH:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            normals = mesh.vertex_normals
    
    # 5. Centrage
    if center:
        vertices = center_and_scale(vertices, target_size)
    
    print(f"\n[OK] Reconstruction terminee")
    print(f"   Vertices : {vertices.shape[0]:,}")
    print(f"   Faces    : {faces.shape[0]:,}")
    
    return vertices, faces, normals


# ============================================================
#  Test standalone
# ============================================================

if __name__ == "__main__":
    print("Test de reconstruction sur volume synthetique...")
    
    # Creer un volume de test : sphere
    volume = np.zeros((64, 64, 64), dtype=np.float64)
    center = np.array([32, 32, 32])
    
    for z in range(64):
        for y in range(64):
            for x in range(64):
                dist = np.sqrt(np.sum((np.array([z, y, x]) - center) ** 2))
                if dist < 20:
                    volume[z, y, x] = 1.0
    
    print(f"  Volume test : {volume.shape}, voxels actifs = {int(volume.sum())}")
    
    # Reconstruction
    vertices, faces, normals = reconstruction_pipeline(
        volume, simplify=False, sigma=1.0
    )
    
    print(f"  Mesh : {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    print("  [OK] Test OK")
