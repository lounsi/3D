# BrainXR — Guide de Configuration Unity

## 1. Creer le projet Unity

1. Ouvrir **Unity Hub**
2. Cliquer **New Project**
3. Selectionner **Unity 2022 LTS** (2022.3.x)
4. Template : **3D (URP)** ou **3D Core**
5. Nom : `BrainXR`
6. Emplacement : `C:\Users\leoma\OneDrive\Bureau\3D\unity\`

## 2. Installer les packages XR

Dans Unity : **Window > Package Manager**

1. **XR Plugin Management** — Cliquer Install
2. **XR Interaction Toolkit** — Cliquer Install
3. **TextMeshPro** — Import TMP Essentials quand demande
4. (VR) **Oculus XR Plugin** — Pour Meta Quest
5. (AR) **ARCore XR Plugin** — Pour Android

## 3. Configurer XR

**Edit > Project Settings > XR Plug-in Management**
- Cocher **Oculus** (pour Meta Quest)
- Ou cocher **ARCore** (pour AR mobile)

## 4. Importer les scripts

Copier les fichiers depuis ce depot :
```
unity/Assets/Scripts/     →  votre projet Assets/Scripts/
unity/Assets/Shaders/     →  votre projet Assets/Shaders/
```

## 5. Creer la scene

### Hierarchie de la scene
```
MainScene
├── BrainModel (empty GameObject)
│   ├── Components : MeshImporter, MeshFilter, MeshRenderer,
│   │                MeshCollider, BrainRotation,
│   │                TransparencyController, SliceViewer,
│   │                XRInteractionController
│   └── Material : BrainXR/VolumeSlice shader
├── Canvas (UI)
│   ├── PanelImport
│   │   ├── BtnImportImages (Button)
│   │   ├── BtnLoadModel (Button)
│   │   ├── TxtSliceCount (TextMeshPro)
│   │   └── BtnNextToTraitement (Button)
│   ├── PanelTraitement
│   │   ├── BtnLancerSegmentation (Button)
│   │   ├── BtnGenererModele (Button)
│   │   ├── ProgressBar (Slider)
│   │   ├── TxtProgress (TextMeshPro)
│   │   ├── TxtLog (TextMeshPro - ScrollRect)
│   │   └── BtnNextToVisu (Button)
│   └── PanelVisualisation
│       ├── SliderTransparence (Slider 0-1)
│       ├── SliderCoupe (Slider 0-1)
│       ├── ToggleAutoRotation (Toggle)
│       ├── BtnResetView (Button)
│       └── TxtMeshInfo (TextMeshPro)
├── UIManager (empty GameObject)
│   └── Component : UIManager.cs
├── Main Camera
├── Directional Light
└── XR Origin (si mode VR)
```

### Assignation des references
1. Selectionner **UIManager** dans la hierarchie
2. Dans l'Inspector, glisser-deposer :
   - Les 3 panels vers `panelImport`, `panelTraitement`, `panelVisualisation`
   - Les boutons, sliders, et textes vers les champs correspondants
   - Le `BrainModel` vers `meshImporter`

3. Selectionner **BrainModel**
4. Creer un **Material** avec le shader `BrainXR/VolumeSlice`
5. Assigner ce material au `MeshRenderer`

## 6. Placer le modele OBJ

Copier le fichier `.obj` genere par le pipeline Python dans :
```
unity/Assets/StreamingAssets/Models/brain.obj
```

## 7. Build pour XR

### Meta Quest (VR)
1. **File > Build Settings** → Switch to **Android**
2. Player Settings : Minimum API Level = **29**
3. XR Plug-in : **Oculus** active
4. Build and Run

### AR Mobile
1. **File > Build Settings** → Switch to **Android**
2. XR Plug-in : **ARCore** active
3. Ajouter **AR Session** + **AR Session Origin** a la scene
4. Build and Run
