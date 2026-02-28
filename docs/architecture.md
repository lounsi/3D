# BrainXR — Architecture Technique

## Vue d'ensemble

```
┌──────────────────────────────────────────────────────────┐
│                    PIPELINE BRAINXR                       │
│                                                          │
│  ┌─────────┐  ┌──────────────┐  ┌───────────────┐       │
│  │ Images  │  │ Pretraitement│  │ Segmentation  │       │
│  │ IRM 2D  │──│ - Normalise  │──│ - U-Net       │       │
│  │ PNG/NII │  │ - Resize     │  │ - Otsu        │       │
│  └─────────┘  │ - Denoise    │  │   (fallback)  │       │
│               │ - CLAHE      │  └──────┬────────┘       │
│               └──────────────┘         │                 │
│                                        ▼                 │
│                              ┌───────────────────┐       │
│                              │ Reconstruction 3D │       │
│                              │ - Build volume    │       │
│                              │ - Smooth          │       │
│                              │ - Marching Cubes  │       │
│                              │ - Simplify mesh   │       │
│                              └────────┬──────────┘       │
│                                       │                  │
│                                       ▼                  │
│                              ┌───────────────────┐       │
│                              │ Export            │       │
│                              │ - OBJ / PLY      │       │
│                              │ - Validation      │       │
│                              └────────┬──────────┘       │
│                                       │                  │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ │
│           Fichier .OBJ sur disque     │                  │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ │
│                                       ▼                  │
│                              ┌───────────────────┐       │
│                              │ Unity XR          │       │
│                              │ - Import mesh     │       │
│                              │ - Visualisation   │       │
│                              │ - Interactions    │       │
│                              │ - Coupe/Transparence     │
│                              └───────────────────┘       │
└──────────────────────────────────────────────────────────┘
```

## Stack technique

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| Segmentation | U-Net (PyTorch) | Standard en imagerie medicale |
| Fallback segmentation | Otsu + morphologie | Robuste sans modele pre-entraine |
| Reconstruction | Marching Cubes (scikit-image) | Algorithme classique, fiable |
| Simplification | Trimesh (decimation quadrique) | Reduction pour temps reel |
| Format echange | Wavefront OBJ | Universel, lisible |
| Moteur XR | Unity 2022 LTS + XR Toolkit | Officiel, multi-plateforme |
| Plan de coupe | Shader HLSL custom | Clipping plane en temps reel |

## Modules Python

| Fichier | Role | Entree | Sortie |
|---------|------|--------|--------|
| `preprocessing.py` | Charger, normaliser, debruiter | Dossier images / NIfTI | Volume numpy 3D |
| `segmentation.py` | Isoler le cerveau | Volume 3D | Masques binaires 3D |
| `reconstruction.py` | Generer le mesh 3D | Masques 3D | Vertices + faces |
| `export_mesh.py` | Exporter en OBJ/PLY | Vertices + faces | Fichier .OBJ |
| `main.py` | Orchestrateur CLI | Arguments | Pipeline complet |

## Scripts Unity C#

| Fichier | Role |
|---------|------|
| `MeshImporter.cs` | Charger le OBJ a runtime |
| `UIManager.cs` | Gestion des 3 ecrans UI |
| `XRInteractionController.cs` | Rotation, zoom, pan (XR + desktop) |
| `SliceViewer.cs` | Plan de coupe interactif |
| `TransparencyController.cs` | Transparence ajustable |
| `BrainRotation.cs` | Auto-rotation du modele |

## Communication Backend ↔ Unity

Le pipeline Python genere un fichier `.OBJ` sur le disque local.
Ce fichier est place dans `unity/Assets/StreamingAssets/Models/`.
Unity le charge a runtime via `MeshImporter.cs`.

**Pas de communication reseau** — tout est local et simple.
