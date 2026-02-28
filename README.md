# 🧠 BrainXR

**Reconstruction 3D du cerveau humain a partir d'images IRM 2D, visualisable en XR (AR/VR)**

Projet academique XR / IA dans le domaine de la sante.

---

## 📁 Structure du projet

```
3D/
├── backend/                    # Pipeline IA (Python)
│   ├── preprocessing.py        # Etape 1 : Pretraitement images
│   ├── segmentation.py         # Etape 2 : Segmentation U-Net / Otsu
│   ├── reconstruction.py       # Etape 3 : Marching Cubes → mesh 3D
│   ├── export_mesh.py          # Etape 4 : Export OBJ / PLY
│   ├── main.py                 # CLI orchestrateur
│   ├── generate_test_data.py   # Generateur de donnees synthetiques
│   └── requirements.txt        # Dependances Python
├── unity/                      # Application Unity (C#)
│   └── Assets/
│       ├── Scripts/            # 6 scripts C#
│       │   ├── MeshImporter.cs
│       │   ├── UIManager.cs
│       │   ├── XRInteractionController.cs
│       │   ├── SliceViewer.cs
│       │   ├── TransparencyController.cs
│       │   └── BrainRotation.cs
│       ├── Shaders/
│       │   └── VolumeSlice.shader
│       └── StreamingAssets/Models/   # Fichiers OBJ generes
└── docs/                       # Documentation
    ├── architecture.md
    ├── pipeline.md
    ├── planning.md
    └── unity_setup.md
```

---

## 🚀 Demarrage rapide

### 1. Backend Python

```bash
cd backend
pip install -r requirements.txt

# Generer des donnees de test
python generate_test_data.py data/input 40

# Lancer le pipeline complet
python main.py --input data/input --output data/output/brain.obj
```

### 2. Unity

Voir [docs/unity_setup.md](docs/unity_setup.md) pour la configuration complete.

1. Creer un projet Unity 2022 LTS
2. Importer les scripts et shaders
3. Copier `brain.obj` dans `StreamingAssets/Models/`
4. Configurer la scene selon le guide

---

## 🔬 Pipeline IA

| Etape | Module | Description |
|-------|--------|-------------|
| 1 | `preprocessing.py` | Normalisation, resize, debruitage, CLAHE |
| 2 | `segmentation.py` | U-Net (deep learning) ou Otsu (fallback) |
| 3 | `reconstruction.py` | Empilement + lissage + Marching Cubes |
| 4 | `export_mesh.py` | Export OBJ/PLY avec validation |

---

## 🎮 Fonctionnalites Unity

- ✅ Import de mesh OBJ a runtime
- ✅ Interface 3 ecrans (Import → Traitement → Visualisation)
- ✅ Rotation / Zoom / Pan (souris + XR controllers)
- ✅ Plan de coupe interactif (SliceViewer)
- ✅ Transparence ajustable
- ✅ Auto-rotation
- ✅ Compatible VR (Meta Quest) et AR (mobile)

---

## 📚 Technologies

| Composant | Technologie |
|-----------|-------------|
| IA / Segmentation | Python, PyTorch, U-Net |
| Reconstruction | scikit-image (Marching Cubes) |
| Visualisation XR | Unity 2022 LTS, XR Interaction Toolkit |
| Shader | HLSL custom (transparence + clipping) |
