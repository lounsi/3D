"""
BrainXR - Generateur de cerveau 3D TRES ATROPHIE (cerveau rouge)
============================================================
Meme pipeline que cerveau.py, mais scanne TOUS les sujets pour
trouver celui avec la PIRE ATROPHIE mesuree (pas juste le CDR).

Le cerveau selectionne devrait ressortir majoritairement rouge/orange
sur la heatmap car ses regions ont perdu le plus de matiere grise
par rapport aux sujets sains.

Difference avec les autres scripts :
  - cerveau.py         → cerveau aleatoire
  - cerveau_malade.py  → cerveau avec le CDR le plus eleve (diagnostic)
  - cerveau_rouge.py   → cerveau avec le plus d'atrophie mesuree (volume)

Usage :
    python cerveau_rouge.py
"""

import os
import sys
import random
import shutil
import subprocess
import numpy as np


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


def find_most_atrophied_subject(oasis):
    """
    Scanne TOUS les sujets et mesure leur atrophie reelle
    pour trouver celui avec le plus de perte de matiere grise.
    
    Contrairement a cerveau_malade.py qui utilise le CDR (diagnostic),
    ici on MESURE directement le volume de chaque cerveau vs la reference.
    
    Returns:
        int: index du sujet le plus atrophie
    """
    from atrophy_analysis import compute_healthy_reference, analyze_atrophy

    n = len(oasis.gray_matter_maps)

    print("\n-> Calcul de la reference saine...")
    reference = compute_healthy_reference(oasis)

    print(f"\n-> Scan de l'atrophie des {n} sujets...")
    print(f"   (cela peut prendre 1-2 minutes)\n")

    results = []
    for i in range(n):
        nifti_path = oasis.gray_matter_maps[i]
        try:
            # Analyse rapide (sans affichage)
            from nilearn import image
            atlas_img = reference["atlas_img"]
            gm_img = image.load_img(nifti_path)
            gm_resampled = image.resample_to_img(gm_img, atlas_img, interpolation='nearest')
            gm_data = gm_resampled.get_fdata()

            total_gm = float(np.sum(gm_data))
            total_mean = reference["mean"][0]
            total_std = reference["std"][0]

            global_atrophy = ((total_mean - total_gm) / total_mean * 100) if total_mean > 0 else 0
            global_atrophy = max(0, global_atrophy)

            # Compter les regions severes
            region_indices = reference["region_indices"]
            atlas_data = reference["atlas_data"]
            severe_count = 0
            for j, ridx in enumerate(region_indices):
                mask = atlas_data == ridx
                vol = float(np.sum(gm_data[mask]))
                mean_h = reference["mean"][1 + j]
                std_h = reference["std"][1 + j]
                z_score = (vol - mean_h) / std_h if std_h > 0 else 0
                if z_score <= -2.0:
                    severe_count += 1

            # CDR pour info
            try:
                cdr = float(oasis.ext_vars.iloc[i].get('cdr', 0))
                if np.isnan(cdr):
                    cdr = -1
            except:
                cdr = -1

            results.append((i, global_atrophy, severe_count, total_gm, cdr))
            print(f"   Sujet #{i+1:>2}/{n} : atrophie={global_atrophy:>5.1f}%  "
                  f"regions severes={severe_count:>2}  "
                  f"CDR={'?' if cdr < 0 else str(cdr)}")

        except Exception as e:
            print(f"   Sujet #{i+1:>2}/{n} : ERREUR ({e})")
            continue

    # Trier : atrophie la plus haute d'abord, puis le plus de regions severes
    results.sort(key=lambda x: (-x[1], -x[2]))

    # Afficher le top 5
    print(f"\n   {'='*60}")
    print(f"   Top 5 des cerveaux les plus ATROPHIES :")
    print(f"   {'Sujet':<10} {'Atrophie':<12} {'Reg. severes':<15} {'CDR':<8}")
    print(f"   {'-'*50}")
    for i, (idx, atrophy, severe, gm, cdr) in enumerate(results[:5]):
        cdr_str = '?' if cdr < 0 else f'{cdr:.1f}'
        marker = " <-- SELECTIONNE" if i == 0 else ""
        print(f"   #{idx+1:<9} {atrophy:<12.1f}% {severe:<15} {cdr_str:<8}{marker}")
    print(f"   {'='*60}")

    chosen = results[0]
    print(f"\n   Sujet selectionne : #{chosen[0] + 1}")
    print(f"   Atrophie globale : {chosen[1]:.1f}%")
    print(f"   Regions severes  : {chosen[2]}")
    print(f"   Volume matiere grise : {chosen[3]:.0f} mm3")

    return chosen[0]


def generate_atrophied_brain(oasis, subject_idx):
    """
    Reconstruit le cerveau 3D du sujet le plus atrophie.
    
    Returns:
        str: chemin du fichier OBJ
    """
    import nibabel as nib
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    chosen = oasis.gray_matter_maps[subject_idx]
    print(f"\n-> Chargement du cerveau le plus atrophie...")

    nii = nib.load(chosen)
    volume = nii.get_fdata().astype(np.float64)
    print(f"   Shape : {volume.shape}, Range : [{volume.min():.2f}, {volume.max():.2f}]")

    # Parametres : seuil bas pour bien voir l'atrophie
    threshold = round(random.uniform(0.15, 0.30), 2)
    sigma = round(random.uniform(0.8, 1.5), 1)
    print(f"\n-> Parametres de reconstruction (optimises pour atrophie) :")
    print(f"   Seuil  : {threshold}")
    print(f"   Lissage: {sigma}")

    volume = np.pad(volume, pad_width=5, mode='constant', constant_values=0)
    volume = gaussian_filter(volume, sigma=sigma)

    print("\n-> Marching Cubes...")
    if volume.max() < threshold:
        threshold = volume.max() * 0.5
        print(f"   Seuil ajuste a {threshold:.2f}")

    vertices, faces, normals, _ = marching_cubes(volume, level=threshold)

    centroid = vertices.mean(axis=0)
    vertices = vertices - centroid
    max_extent = np.max(np.abs(vertices))
    if max_extent > 0:
        vertices = vertices / max_extent

    print(f"   Vertices : {vertices.shape[0]:,}")
    print(f"   Faces    : {faces.shape[0]:,}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "brain.obj")

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
        f.write("# BrainXR - Cerveau 3D TRES ATROPHIE (rouge)\n")
        f.write(f"# Sujet OASIS #{subject_idx + 1}, seuil={threshold}, sigma={sigma}\n")
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

    return output_file


def run_ai_analysis(oasis, subject_idx):
    """Lance l'analyse IA complete sur le sujet."""
    from alzheimer_classifier import full_pipeline, classify_brain
    from atrophy_analysis import compute_healthy_reference, analyze_atrophy, generate_heatmap_obj
    from brain_report import generate_report, extract_demographic

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data", "output")
    nifti_path = oasis.gray_matter_maps[subject_idx]

    print("\n" + "=" * 60)
    print("    ANALYSE IA DU CERVEAU ATROPHIE")
    print("=" * 60)

    # --- 1. Classification Alzheimer ---
    print("\n" + "-" * 40)
    print("  ETAPE IA 1 : Classification Alzheimer")
    print("-" * 40)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        shutil.copy2(obj_file, os.path.join(unity_dest_dir, "brain.obj"))
        print(f"-> Copie brain.obj dans Unity : OK")

        heatmap = os.path.join(output_dir, "brain_heatmap.obj")
        if os.path.exists(heatmap):
            shutil.copy2(heatmap, os.path.join(unity_dest_dir, "brain_heatmap.obj"))
            print(f"-> Copie brain_heatmap.obj dans Unity : OK")

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
    print("  BrainXR - Cerveau 3D ROUGE (pire atrophie)")
    print("  Ce script mesure l'atrophie de TOUS les sujets")
    print("  et selectionne le plus atrophie !")
    print("=" * 60)

    # Etape 1 : Telecharger OASIS
    from nilearn import datasets
    print("\n-> Telechargement des cerveaux OASIS...")
    oasis = datasets.fetch_oasis_vbm(n_subjects=30)

    # Etape 2 : Scanner tous les sujets pour trouver le plus atrophie
    subject_idx = find_most_atrophied_subject(oasis)

    # Etape 3 : Generer le cerveau 3D
    obj_file = generate_atrophied_brain(oasis, subject_idx)

    # Etape 4 : Analyse IA
    report = run_ai_analysis(oasis, subject_idx)

    # Etape 5 : Copier dans Unity
    copy_to_unity(obj_file)

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE CERVEAU ROUGE TERMINE !")
    print(f"  ")
    print(f"  Fichiers generes :")
    print(f"    - brain.obj          (modele 3D atrophie)")
    print(f"    - brain_heatmap.obj  (heatmap rouge)")
    print(f"    - brain_report.json  (rapport diagnostic)")
    print(f"  ")
    print(f"  Diagnostic : {report['classification']['prediction']}"
          f" ({report['classification']['confidence']}%)")
    print(f"  Atrophie   : {report['atrophy']['global_atrophy_percent']}%")
    print(f"  Regions severes : {report['atrophy']['status_counts']['severe']}")
    print(f"  ")
    print(f"  Pour un cerveau ALEATOIRE : python cerveau.py")
    print(f"  Pour un cerveau MALADE    : python cerveau_malade.py")
    print(f"  Pour un cerveau ROUGE     : python cerveau_rouge.py")
    print(f"  Pour voir dans Unity      : appuyez sur Play.")
    print(f"{'=' * 60}")
