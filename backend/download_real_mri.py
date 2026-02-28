"""
BrainXR - Generateur de cerveau 3D aleatoire + Analyse IA
============================================================
Telecharge un cerveau aleatoire depuis OASIS (30 sujets),
reconstruit un modele 3D, puis lance l'analyse IA :
- Classification Alzheimer (Sain / Declin leger / Alzheimer)
- Analyse d'atrophie (comparaison aux sujets sains)
- Export rapport JSON + mesh heatmap colorise

Usage :
    python download_real_mri.py
"""

import os
import sys
import random
import shutil
import subprocess


def ensure_packages():
    """Installe les packages si necessaire."""
    for pkg in ["nilearn", "nibabel", "scikit-image", "trimesh", "scipy"]:
        try:
            __import__(pkg.replace("-", "_").replace("scikit_image", "skimage"))
        except ImportError:
            print(f"Installation de {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL
            )


def generate_random_brain():
    """
    Telecharge un cerveau aleatoire depuis OASIS et
    le reconstruit directement en 3D avec Marching Cubes.
    
    Returns:
        tuple (output_file, oasis_dataset, subject_index)
    """
    from nilearn import datasets
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    # --- Telecharger un sujet aleatoire ---
    print("\n-> Telechargement des cerveaux OASIS...")
    oasis = datasets.fetch_oasis_vbm(n_subjects=30)

    idx = random.randint(0, len(oasis.gray_matter_maps) - 1)
    chosen = oasis.gray_matter_maps[idx]
    print(f"-> Cerveau choisi : sujet #{idx + 1} / {len(oasis.gray_matter_maps)}")

    # --- Charger le volume ---
    nii = nib.load(chosen)
    volume = nii.get_fdata().astype(np.float64)
    print(f"   Shape : {volume.shape}, Range : [{volume.min():.2f}, {volume.max():.2f}]")

    # --- Parametres aleatoires ---
    threshold = round(random.uniform(0.15, 0.45), 2)
    sigma = round(random.uniform(0.8, 2.0), 1)
    print(f"\n-> Parametres de reconstruction :")
    print(f"   Seuil  : {threshold} (bas=gros cerveau, haut=petit)")
    print(f"   Lissage: {sigma}")

    # --- Padding 3D (evite les carres plats) ---
    volume = np.pad(volume, pad_width=5, mode='constant', constant_values=0)

    # --- Lissage ---
    volume = gaussian_filter(volume, sigma=sigma)

    # --- Marching Cubes ---
    print("\n-> Marching Cubes...")
    if volume.max() < threshold:
        threshold = volume.max() * 0.5
        print(f"   Seuil ajuste a {threshold:.2f}")

    vertices, faces, normals, _ = marching_cubes(volume, level=threshold)

    # --- Centrer et mettre a l'echelle ---
    centroid = vertices.mean(axis=0)
    vertices = vertices - centroid
    max_extent = np.max(np.abs(vertices))
    if max_extent > 0:
        vertices = vertices / max_extent

    print(f"   Vertices : {vertices.shape[0]:,}")
    print(f"   Faces    : {faces.shape[0]:,}")

    # --- Export OBJ ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "brain.obj")

    # Calcul des normales
    import numpy as np
    mesh_normals = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        fn = np.cross(v1 - v0, v2 - v0)
        mesh_normals[face[0]] += fn
        mesh_normals[face[1]] += fn
        mesh_normals[face[2]] += fn
    norms = np.linalg.norm(mesh_normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    mesh_normals = mesh_normals / norms

    with open(output_file, 'w') as f:
        f.write("# BrainXR - Cerveau 3D\n")
        f.write(f"# Sujet OASIS #{idx + 1}, seuil={threshold}, sigma={sigma}\n")
        f.write(f"# Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}\n")
        f.write("o BrainModel\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for n in mesh_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n")
        for face in faces:
            i0, i1, i2 = face[0]+1, face[1]+1, face[2]+1
            f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n-> OBJ exporte : {output_file} ({size_mb:.1f} MB)")

    return output_file, oasis, idx


def run_ai_analysis(oasis, subject_idx):
    """
    Lance l'analyse IA complete sur le sujet choisi.
    
    1. Classification Alzheimer
    2. Analyse d'atrophie
    3. Generation du rapport JSON
    4. Generation du mesh heatmap
    
    Args:
        oasis: dataset OASIS
        subject_idx: index du sujet
        
    Returns:
        dict du rapport
    """
    import numpy as np
    from alzheimer_classifier import full_pipeline, classify_brain
    from atrophy_analysis import compute_healthy_reference, analyze_atrophy, generate_heatmap_obj
    from brain_report import generate_report, extract_demographic

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data", "output")
    nifti_path = oasis.gray_matter_maps[subject_idx]

    print("\n" + "=" * 60)
    print("    ANALYSE IA DU CERVEAU")
    print("=" * 60)

    # --- 1. Classification Alzheimer ---
    print("\n" + "-" * 40)
    print("  ETAPE IA 1 : Classification Alzheimer")
    print("-" * 40)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Verifier si un modele entraine existe deja
    model_path = os.path.join(script_dir, "data", "models", "alzheimer_classifier.pth")

    if os.path.exists(model_path):
        print(f"-> Modele pre-entraine trouve : {model_path}")
        classification_result = classify_brain(
            nifti_path, model_path=model_path, device=device
        )
    else:
        print("-> Pas de modele pre-entraine, entrainement...")
        model, scaler_mean, scaler_std, _ = full_pipeline(
            n_subjects=30, device=device
        )
        classification_result = classify_brain(
            nifti_path, model=model,
            scaler_mean=scaler_mean, scaler_std=scaler_std,
            device=device
        )

    # --- 2. Analyse d'atrophie ---
    print("\n" + "-" * 40)
    print("  ETAPE IA 2 : Analyse d'atrophie")
    print("-" * 40)

    reference = compute_healthy_reference(oasis)
    atrophy_result = analyze_atrophy(nifti_path, reference)

    # --- 3. Mesh heatmap ---
    print("\n" + "-" * 40)
    print("  ETAPE IA 3 : Mesh heatmap")
    print("-" * 40)

    heatmap_path = os.path.join(output_dir, "brain_heatmap.obj")
    generate_heatmap_obj(nifti_path, reference, heatmap_path)

    # --- 4. Rapport JSON ---
    print("\n" + "-" * 40)
    print("  ETAPE IA 4 : Rapport JSON")
    print("-" * 40)

    demographic = extract_demographic(oasis, subject_idx)
    report_path = os.path.join(output_dir, "brain_report.json")

    report = generate_report(
        subject_id=subject_idx,
        classification_result=classification_result,
        atrophy_result=atrophy_result,
        demographic=demographic,
        output_path=report_path
    )

    return report


def copy_to_unity(obj_file):
    """Copie les fichiers generes dans le projet Unity."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data", "output")
    unity_dest_dir = os.path.join(
        script_dir, "..", "unity", "BrainXR",
        "Assets", "StreamingAssets", "Models"
    )

    if os.path.exists(unity_dest_dir):
        # Copier brain.obj
        shutil.copy2(obj_file, os.path.join(unity_dest_dir, "brain.obj"))
        print(f"-> Copie brain.obj dans Unity : OK")

        # Copier brain_heatmap.obj si il existe
        heatmap = os.path.join(output_dir, "brain_heatmap.obj")
        if os.path.exists(heatmap):
            shutil.copy2(heatmap, os.path.join(unity_dest_dir, "brain_heatmap.obj"))
            print(f"-> Copie brain_heatmap.obj dans Unity : OK")

        # Copier brain_report.json
        report = os.path.join(output_dir, "brain_report.json")
        if os.path.exists(report):
            shutil.copy2(report, os.path.join(unity_dest_dir, "brain_report.json"))
            print(f"-> Copie brain_report.json dans Unity : OK")
    else:
        print(f"-> Dossier Unity non trouve, copie manuelle necessaire")
        print(f"   Copiez les fichiers de {output_dir} vers")
        print(f"   Unity > Assets > StreamingAssets > Models")


if __name__ == "__main__":
    ensure_packages()

    print("=" * 60)
    print("  BrainXR - Generateur de cerveau 3D + Analyse IA")
    print("  Chaque lancement = un cerveau different !")
    print("=" * 60)

    # Etape 1 : Generer le cerveau 3D
    obj_file, oasis, subject_idx = generate_random_brain()

    # Etape 2 : Analyse IA
    report = run_ai_analysis(oasis, subject_idx)

    # Etape 3 : Copier dans Unity
    copy_to_unity(obj_file)

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLET TERMINE !")
    print(f"  ")
    print(f"  Fichiers generes :")
    print(f"    - brain.obj          (modele 3D)")
    print(f"    - brain_heatmap.obj  (heatmap atrophie)")
    print(f"    - brain_report.json  (rapport diagnostic)")
    print(f"  ")
    print(f"  Diagnostic : {report['classification']['prediction']}"
          f" ({report['classification']['confidence']}%)")
    print(f"  Atrophie   : {report['atrophy']['global_atrophy_percent']}%")
    print(f"  ")
    print(f"  Pour un AUTRE cerveau : relancez ce script.")
    print(f"  Pour voir dans Unity  : appuyez sur Play.")
    print(f"{'=' * 60}")
