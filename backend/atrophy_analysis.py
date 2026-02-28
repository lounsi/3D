"""
BrainXR - Analyse d'Atrophie Cerebrale
========================================
Compare le volume de matiere grise d'un sujet
aux sujets sains de la cohorte OASIS.

Identifie les zones atrophiees et genere un mesh
OBJ colorise (heatmap : vert=normal, rouge=atrophie).

Usage :
    python atrophy_analysis.py
"""

import os
import sys
import numpy as np
import json

# ============================================================
#  Calcul des statistiques de reference (sujets sains)
# ============================================================

def compute_healthy_reference(oasis_dataset):
    """
    Calcule les volumes moyens par region pour les sujets sains (CDR=0).

    Args:
        oasis_dataset: dataset retourne par fetch_oasis_vbm()

    Returns:
        dict avec mean, std par region et region_names
    """
    from nilearn import datasets, image
    import nibabel as nib

    print("\n-> Calcul des statistiques de reference (sujets sains)...")

    # Identifier les sujets sains (CDR = 0)
    cdr_values = oasis_dataset.ext_vars['cdr'].values
    healthy_indices = [i for i, cdr in enumerate(cdr_values)
                       if not np.isnan(cdr) and cdr == 0]

    print(f"   Sujets sains (CDR=0) : {len(healthy_indices)}")

    if len(healthy_indices) < 3:
        print("   /!\\ Pas assez de sujets sains pour une reference fiable")

    # Charger l'atlas
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = image.load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    labels = atlas.labels
    region_indices = list(range(1, len(labels)))
    region_names = [labels[i] for i in region_indices]

    n_regions = len(region_indices)
    n_healthy = len(healthy_indices)

    # Matrice volumes : (n_healthy, n_regions + 1)
    volumes = np.zeros((n_healthy, n_regions + 1))  # +1 pour le total

    for k, idx in enumerate(healthy_indices):
        gm_path = oasis_dataset.gray_matter_maps[idx]
        gm_img = image.load_img(gm_path)
        gm_resampled = image.resample_to_img(gm_img, atlas_img, interpolation='nearest')
        gm_data = gm_resampled.get_fdata()

        # Volume total
        volumes[k, 0] = float(np.sum(gm_data))

        # Volume par region
        for j, ridx in enumerate(region_indices):
            mask = atlas_data == ridx
            volumes[k, 1 + j] = float(np.sum(gm_data[mask]))

        if (k + 1) % 5 == 0 or k == n_healthy - 1:
            print(f"   Reference {k + 1}/{n_healthy} traitee")

    reference = {
        "region_names": region_names,
        "region_indices": region_indices,
        "mean": volumes.mean(axis=0),
        "std": volumes.std(axis=0),
        "n_healthy": n_healthy,
        "atlas_img": atlas_img,
        "atlas_data": atlas_data
    }

    return reference


# ============================================================
#  Analyse d'atrophie d'un sujet
# ============================================================

def analyze_atrophy(nifti_path, reference, subject_info=None):
    """
    Analyse l'atrophie d'un sujet par rapport a la reference saine.

    Args:
        nifti_path: chemin vers le fichier NIfTI du sujet
        reference: dict retourne par compute_healthy_reference()
        subject_info: dict optionnel avec age, sex, etc.

    Returns:
        dict avec les resultats d'analyse
    """
    from nilearn import image

    print(f"\n-> Analyse d'atrophie : {os.path.basename(nifti_path)}")

    atlas_img = reference["atlas_img"]
    atlas_data = reference["atlas_data"]
    region_names = reference["region_names"]
    region_indices = reference["region_indices"]
    ref_mean = reference["mean"]
    ref_std = reference["std"]

    # Charger le sujet
    gm_img = image.load_img(nifti_path)
    gm_resampled = image.resample_to_img(gm_img, atlas_img, interpolation='nearest')
    gm_data = gm_resampled.get_fdata()

    n_regions = len(region_indices)

    # Volume total
    total_gm = float(np.sum(gm_data))
    total_mean = ref_mean[0]
    total_std = ref_std[0]
    total_z = (total_gm - total_mean) / total_std if total_std > 0 else 0

    global_atrophy = ((total_mean - total_gm) / total_mean * 100) if total_mean > 0 else 0

    # Analyse par region
    regions = []
    for j, ridx in enumerate(region_indices):
        mask = atlas_data == ridx
        vol = float(np.sum(gm_data[mask]))
        mean_h = ref_mean[1 + j]
        std_h = ref_std[1 + j]

        z_score = (vol - mean_h) / std_h if std_h > 0 else 0
        deviation = ((vol - mean_h) / mean_h * 100) if mean_h > 0 else 0

        # Statut
        if z_score <= -2.0:
            status = "severe"
        elif z_score <= -1.5:
            status = "moderate"
        elif z_score <= -1.0:
            status = "mild"
        else:
            status = "normal"

        regions.append({
            "name": region_names[j],
            "volume": round(vol, 1),
            "mean_healthy": round(mean_h, 1),
            "std_healthy": round(std_h, 1),
            "z_score": round(z_score, 2),
            "deviation_percent": round(deviation, 1),
            "status": status
        })

    # Trier par z-score (les plus atrophiees en premier)
    regions.sort(key=lambda r: r["z_score"])

    # Top 5 des plus atrophiees
    most_atrophied = [r["name"] for r in regions[:5] if r["z_score"] < -1.0]

    # Compter par statut
    status_counts = {
        "normal": sum(1 for r in regions if r["status"] == "normal"),
        "mild": sum(1 for r in regions if r["status"] == "mild"),
        "moderate": sum(1 for r in regions if r["status"] == "moderate"),
        "severe": sum(1 for r in regions if r["status"] == "severe"),
    }

    result = {
        "global_atrophy_percent": round(max(0, global_atrophy), 1),
        "total_gray_matter_volume": round(total_gm, 1),
        "total_z_score": round(total_z, 2),
        "regions": regions,
        "most_atrophied": most_atrophied,
        "status_counts": status_counts,
        "reference_subjects": reference["n_healthy"]
    }

    print(f"   Atrophie globale : {result['global_atrophy_percent']}%")
    print(f"   Regions normales : {status_counts['normal']}")
    print(f"   Regions legeres  : {status_counts['mild']}")
    print(f"   Regions moderees : {status_counts['moderate']}")
    print(f"   Regions severes  : {status_counts['severe']}")

    if most_atrophied:
        print(f"   Plus atrophiees  : {', '.join(most_atrophied)}")

    return result


# ============================================================
#  Generation du mesh OBJ heatmap colore
# ============================================================

def generate_heatmap_obj(nifti_path, reference, output_path):
    """
    Genere un mesh OBJ colore representant l'atrophie.
    
    Vert = normal, Jaune = leger, Orange = modere, Rouge = severe.

    Args:
        nifti_path: chemin vers le fichier NIfTI
        reference: dict de reference
        output_path: chemin de sortie du fichier OBJ

    Returns:
        chemin du fichier cree
    """
    from nilearn import image
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    print(f"\n-> Generation du mesh heatmap...")

    atlas_img = reference["atlas_img"]
    atlas_data = reference["atlas_data"]
    region_indices = reference["region_indices"]
    ref_mean = reference["mean"]
    ref_std = reference["std"]

    # Charger le sujet
    gm_img = image.load_img(nifti_path)
    gm_resampled = image.resample_to_img(gm_img, atlas_img, interpolation='nearest')
    gm_data = gm_resampled.get_fdata()

    # Creer un volume de z-scores par voxel
    z_score_volume = np.zeros_like(gm_data)

    for j, ridx in enumerate(region_indices):
        mask = atlas_data == ridx
        vol = float(np.sum(gm_data[mask]))
        mean_h = ref_mean[1 + j]
        std_h = ref_std[1 + j]

        if std_h > 0:
            z = (vol - mean_h) / std_h
        else:
            z = 0

        z_score_volume[mask] = z

    # Reconstruction 3D du cerveau (meme pipeline que download_real_mri.py)
    volume = gm_data.copy()

    # Padding
    volume = np.pad(volume, pad_width=5, mode='constant', constant_values=0)
    z_score_volume = np.pad(z_score_volume, pad_width=5, mode='constant', constant_values=0)

    # Lissage
    sigma = 1.2
    volume_smooth = gaussian_filter(volume, sigma=sigma)

    # Seuil adaptatif
    threshold = 0.2
    if volume_smooth.max() < threshold:
        threshold = volume_smooth.max() * 0.5

    # Marching Cubes
    try:
        vertices, faces, normals, _ = marching_cubes(volume_smooth, level=threshold)
    except ValueError:
        print("   /!\\ Impossible de generer le mesh heatmap (volume trop faible)")
        return None

    # Centrer et normaliser
    centroid = vertices.mean(axis=0)
    vertices = vertices - centroid
    max_extent = np.max(np.abs(vertices))
    if max_extent > 0:
        vertices = vertices / max_extent

    print(f"   Heatmap mesh : {vertices.shape[0]:,} vertices, {faces.shape[0]:,} faces")

    # Assigner une couleur a chaque vertex basee sur le z-score
    vertex_colors = np.zeros((vertices.shape[0], 3))

    for i, v in enumerate(vertices):
        # Coordonnees dans le volume original (avant centrage/scaling)
        orig = v * max_extent + centroid
        ix = int(np.clip(orig[0], 0, z_score_volume.shape[0] - 1))
        iy = int(np.clip(orig[1], 0, z_score_volume.shape[1] - 1))
        iz = int(np.clip(orig[2], 0, z_score_volume.shape[2] - 1))

        z = z_score_volume[ix, iy, iz]

        # Palette : vert (normal) -> jaune -> orange -> rouge (severe)
        if z >= -0.5:
            # Normal -> Vert
            vertex_colors[i] = [0.2, 0.8, 0.3]
        elif z >= -1.0:
            # Leger -> Jaune-vert
            t = (z + 0.5) / (-0.5)  # 0 to 1
            vertex_colors[i] = [0.2 + 0.6 * t, 0.8 - 0.2 * t, 0.3 - 0.2 * t]
        elif z >= -1.5:
            # Modere -> Jaune-Orange
            t = (z + 1.0) / (-0.5)
            vertex_colors[i] = [0.8 + 0.15 * t, 0.6 - 0.25 * t, 0.1 - 0.05 * t]
        elif z >= -2.0:
            # Modere-Severe -> Orange
            t = (z + 1.5) / (-0.5)
            vertex_colors[i] = [0.95 + 0.05 * t, 0.35 - 0.2 * t, 0.05 - 0.02 * t]
        else:
            # Severe -> Rouge
            vertex_colors[i] = [1.0, 0.15, 0.05]

    # Ecrire le OBJ avec couleurs (extension non-standard mais supportee)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Recalculer les normales
    mesh_normals = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        fn = np.cross(v1 - v0, v2 - v0)
        mesh_normals[face[0]] += fn
        mesh_normals[face[1]] += fn
        mesh_normals[face[2]] += fn
    norms_mag = np.linalg.norm(mesh_normals, axis=1, keepdims=True)
    norms_mag[norms_mag < 1e-8] = 1.0
    mesh_normals = mesh_normals / norms_mag

    # Ecrire le fichier OBJ avec couleurs vertex (format: v x y z r g b)
    with open(output_path, 'w') as f:
        f.write("# BrainXR - Heatmap d'atrophie 3D\n")
        f.write(f"# Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}\n")
        f.write("# Couleurs: vert=normal, jaune=leger, orange=modere, rouge=severe\n")
        f.write("o BrainHeatmap\n\n")

        # Vertices avec couleurs (v x y z r g b)
        for v, c in zip(vertices, vertex_colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                    f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")

        f.write("\n")

        # Normales
        for n in mesh_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        f.write("\n")

        # Faces
        for face in faces:
            i0, i1, i2 = face[0] + 1, face[1] + 1, face[2] + 1
            f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   Heatmap OBJ : {output_path} ({size_mb:.1f} MB)")

    return output_path


# ============================================================
#  Test standalone
# ============================================================

if __name__ == "__main__":
    import random
    from nilearn import datasets

    print("=" * 50)
    print("  ANALYSE D'ATROPHIE CEREBRALE")
    print("=" * 50)

    # Telecharger OASIS
    print("\n-> Telechargement OASIS VBM...")
    oasis = datasets.fetch_oasis_vbm(n_subjects=30)

    # Calculer la reference
    reference = compute_healthy_reference(oasis)

    # Analyser un sujet aleatoire
    idx = random.randint(0, len(oasis.gray_matter_maps) - 1)
    test_path = oasis.gray_matter_maps[idx]
    cdr = oasis.ext_vars['cdr'].iloc[idx]

    print(f"\n{'=' * 50}")
    print(f"  SUJET #{idx + 1} (CDR = {cdr})")
    print(f"{'=' * 50}")

    result = analyze_atrophy(test_path, reference)

    # Afficher les 10 premieres regions
    print(f"\n-> Top 10 regions (par z-score) :")
    for r in result["regions"][:10]:
        print(f"   {r['name']:<30} z={r['z_score']:>6.2f}  "
              f"dev={r['deviation_percent']:>6.1f}%  [{r['status']}]")

    # Generer la heatmap
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "data", "output", "brain_heatmap.obj")
    generate_heatmap_obj(test_path, reference, output_path)

    print(f"\n{'=' * 50}")
    print(f"  ANALYSE TERMINEE")
    print(f"{'=' * 50}")
