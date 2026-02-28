"""
BrainXR - Generateur de rapport JSON
======================================
Genere un rapport complet au format JSON
combinant classification Alzheimer + analyse d'atrophie.

Le JSON est lu par Unity (DiagnosticPanel.cs) pour l'affichage.
"""

import os
import json
import numpy as np
from datetime import datetime


def generate_report(
    subject_id,
    classification_result,
    atrophy_result,
    demographic=None,
    mesh_file="brain.obj",
    heatmap_file="brain_heatmap.obj",
    output_path=None
):
    """
    Genere le rapport JSON complet.

    Args:
        subject_id: index du sujet dans OASIS
        classification_result: dict retourne par classify_brain()
        atrophy_result: dict retourne par analyze_atrophy()
        demographic: dict optionnel {age, sex, mmse, cdr}
        mesh_file: nom du fichier OBJ principal
        heatmap_file: nom du fichier OBJ heatmap
        output_path: chemin de sortie du JSON

    Returns:
        dict du rapport complet
    """
    print("\n-> Generation du rapport JSON...")

    # Simplifier les regions pour le JSON (top 10)
    top_regions = []
    for r in atrophy_result["regions"][:10]:
        top_regions.append({
            "name": r["name"],
            "z_score": r["z_score"],
            "status": r["status"],
            "deviation_percent": r["deviation_percent"]
        })

    report = {
        "subject_id": int(subject_id) + 1,
        "timestamp": datetime.now().isoformat(),
        "demographic": demographic or {},
        "classification": classification_result,
        "atrophy": {
            "global_atrophy_percent": atrophy_result["global_atrophy_percent"],
            "total_gray_matter_volume": atrophy_result["total_gray_matter_volume"],
            "total_z_score": atrophy_result["total_z_score"],
            "regions": top_regions,
            "most_atrophied": atrophy_result["most_atrophied"],
            "status_counts": atrophy_result["status_counts"],
            "reference_subjects": atrophy_result["reference_subjects"]
        },
        "mesh_file": mesh_file,
        "heatmap_mesh_file": heatmap_file
    }

    # Sauvegarder
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "data", "output", "brain_report.json")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"   Rapport JSON : {output_path} ({size_kb:.1f} KB)")

    # Affichage resume
    print(f"\n   === RESUME DU DIAGNOSTIC ===")
    print(f"   Sujet        : #{report['subject_id']}")
    if demographic:
        print(f"   Age          : {demographic.get('age', 'N/A')}")
        print(f"   CDR reel     : {demographic.get('cdr', 'N/A')}")
    print(f"   Diagnostic   : {classification_result['prediction']} "
          f"({classification_result['confidence']}%)")
    print(f"   Atrophie     : {atrophy_result['global_atrophy_percent']}%")
    if atrophy_result['most_atrophied']:
        print(f"   Zones a risque: {', '.join(atrophy_result['most_atrophied'][:3])}")

    return report


def extract_demographic(oasis_dataset, subject_idx):
    """
    Extrait les informations demographiques d'un sujet OASIS.

    ext_vars est un DataFrame pandas avec colonnes :
    id, mf, hand, age, educ, ses, mmse, cdr, etiv, nwbv, asf, delay
    """
    row = oasis_dataset.ext_vars.iloc[subject_idx]

    def safe_float(val, decimals=1):
        try:
            v = float(val)
            return round(v, decimals) if not np.isnan(v) else None
        except (ValueError, TypeError):
            return None

    demographic = {
        "age": safe_float(row.get('age')),
        "sex": str(row.get('mf', 'N/A')),
        "education": safe_float(row.get('educ')),
        "socioeconomic": safe_float(row.get('ses')),
        "mmse": safe_float(row.get('mmse')),
        "cdr": safe_float(row.get('cdr'), 2)
    }

    return demographic
