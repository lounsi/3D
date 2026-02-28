# BrainXR — Pipeline IA Detail

## Etape 1 : Pretraitement (`preprocessing.py`)

### Objectif
Transformer les images IRM brutes en volume 3D normalise et propre.

### Sous-etapes
1. **Chargement** — Auto-detection du format (PNG, NIfTI, DICOM)
2. **Normalisation** — Intensites ramenees a [0, 1]
3. **Redimensionnement** — Toutes les slices a 256x256
4. **Debruitage** — Filtre gaussien (σ=1.0)
5. **CLAHE** — Amelioration du contraste local

### Formats supportes
- Images 2D : PNG, JPG, BMP, TIFF
- Volume : NIfTI (.nii, .nii.gz)
- Serie : DICOM (.dcm) via SimpleITK

---

## Etape 2 : Segmentation (`segmentation.py`)

### Objectif
Isoler le cerveau du fond et du crane.

### Methode principale : U-Net
- Architecture encoder-decoder avec skip connections
- 4 niveaux de profondeur, features = [32, 64, 128, 256]
- Entree : 1 canal (niveaux de gris), Sortie : 1 canal (masque)
- Activation finale : sigmoide → seuil 0.5

### Methode fallback : Otsu
Si aucun modele pre-entraine n'est disponible :
1. Seuillage Otsu automatique
2. Fermeture morphologique (comble les trous)
3. Ouverture morphologique (supprime le bruit)
4. Selection du plus grand composant connexe

### Pourquoi U-Net ?
- Standard en segmentation d'images medicales depuis 2015
- Architecture legere, entraînable sur petits datasets
- Skip connections preservent les details spatiaux

---

## Etape 3 : Reconstruction 3D (`reconstruction.py`)

### Objectif
Convertir les masques 2D en surface 3D maillée.

### Algorithme : Marching Cubes
1. **Empilement** — Les masques 2D forment un volume binaire 3D
2. **Lissage** — Filtre gaussien 3D (σ=1.5) pour surface lisse
3. **Marching Cubes** — Parcourt chaque voxel, genere des triangles a l'isosurface (seuil=0.5)
4. **Simplification** — Decimation quadrique pour reduire la complexite (50% par defaut)
5. **Centrage** — Modele centre a l'origine, mis a l'echelle

### Pourquoi Marching Cubes ?
- Algorithme classique publie en 1987 par Lorensen & Cline
- Implementation fiable dans scikit-image
- Resultat directement utilisable comme mesh polygonal

---

## Etape 4 : Export (`export_mesh.py`)

### Formats
- **OBJ** (Wavefront) — Choix par defaut, universellement supporte
- **PLY** — Format alternatif

### Contenu du fichier OBJ
```
o BrainModel
v x y z          # Sommets
vn nx ny nz      # Normales
f v//vn v//vn v//vn  # Faces triangulaires
```

### Validation pre-export
- Verification des indices de faces
- Detection des faces degenerees
- Verification NaN / Inf
- Calcul bounding box

---

## Utilisation CLI

```bash
# Avec images PNG
python main.py --input data/input --output data/output/brain.obj

# Avec fichier NIfTI
python main.py --input brain.nii.gz --output brain.obj

# Options avancees
python main.py --input data/input --output brain.ply \
  --format ply --size 256 --threshold 0.4 \
  --method otsu --no-simplify
```

---

## Datasets publics recommandes

| Dataset | Description | Lien |
|---------|-------------|------|
| IXI | 600 cerveaux IRM sains | brain-development.org/ixi-dataset |
| OASIS | Cerveaux normaux et Alzheimer | oasis-brains.org |
| BrainWeb | Cerveaux synthetiques (phantom) | brainweb.bic.mni.mcgill.ca |
