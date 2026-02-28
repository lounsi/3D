"""
BrainXR - CLI Orchestrator
============================
Point d'entree principal du pipeline IA.
Enchaine : Pretraitement -> Segmentation -> Reconstruction -> Export
"""

import argparse
import os
import sys
import time
import numpy as np

from preprocessing import preprocess_pipeline
from segmentation import segment_volume
from reconstruction import reconstruction_pipeline
from export_mesh import export_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="BrainXR - Pipeline de reconstruction 3D du cerveau",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python main.py --input data/input --output data/output/brain.obj
  python main.py --input brain.nii.gz --output brain.ply --format ply
  python main.py --input data/input --output brain.obj --no-gpu --no-simplify
        """
    )
    
    parser.add_argument("--input", "-i", required=True,
                        help="Dossier d'images 2D ou fichier NIfTI")
    parser.add_argument("--output", "-o", required=True,
                        help="Chemin de sortie du mesh (.obj ou .ply)")
    parser.add_argument("--format", "-f", default="obj", choices=["obj", "ply"],
                        help="Format d'export (defaut: obj)")
    parser.add_argument("--size", "-s", type=int, default=256,
                        help="Taille cible des slices (defaut: 256)")
    parser.add_argument("--sigma-preprocess", type=float, default=1.0,
                        help="Sigma debruitage (defaut: 1.0)")
    parser.add_argument("--sigma-smooth", type=float, default=1.5,
                        help="Sigma lissage 3D (defaut: 1.5)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Seuil Marching Cubes (defaut: 0.5)")
    parser.add_argument("--method", "-m", default="auto",
                        choices=["auto", "unet", "otsu"],
                        help="Methode de segmentation (defaut: auto)")
    parser.add_argument("--weights", "-w", default=None,
                        help="Chemin poids U-Net (.pth)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Forcer l'utilisation du CPU")
    parser.add_argument("--no-simplify", action="store_true",
                        help="Desactiver la simplification du mesh")
    parser.add_argument("--simplify-ratio", type=float, default=0.5,
                        help="Ratio de simplification (defaut: 0.5)")
    parser.add_argument("--no-clahe", action="store_true",
                        help="Desactiver CLAHE")
    
    args = parser.parse_args()
    
    # Device
    import torch
    if args.no_gpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    
    print("=" * 60)
    print("    BrainXR - Pipeline de Reconstruction 3D")
    print("=" * 60)
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Format : {args.format.upper()}")
    print(f"  Device : {device}")
    print(f"  Method : {args.method}")
    print("=" * 60)
    
    start = time.time()
    
    # Etape 1 - Pretraitement
    volume = preprocess_pipeline(
        args.input,
        target_size=(args.size, args.size),
        sigma=args.sigma_preprocess,
        use_clahe=not args.no_clahe
    )
    
    # Etape 2 - Segmentation
    masks = segment_volume(
        volume,
        weights_path=args.weights,
        device=device,
        method=args.method
    )
    
    # Etape 3 - Reconstruction
    vertices, faces, normals = reconstruction_pipeline(
        masks,
        sigma=args.sigma_smooth,
        threshold=args.threshold,
        simplify=not args.no_simplify,
        simplify_ratio=args.simplify_ratio
    )
    
    # Etape 4 - Export
    filepath = export_pipeline(
        vertices, faces, args.output,
        format=args.format, normals=normals
    )
    
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  PIPELINE TERMINE en {elapsed:.1f}s")
    print(f"  Fichier : {filepath}")
    print("=" * 60)


if __name__ == "__main__":
    main()
