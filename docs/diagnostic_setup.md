# Diagnostic IA — Guide de setup Unity

## Prérequis
Avant de commencer, lancez le pipeline Python :
```bash
cd backend
python download_real_mri.py
```
Cela génère 3 fichiers dans `backend/data/output/` :
- `brain.obj` — Modèle 3D du cerveau
- `brain_heatmap.obj` — Modèle coloré (atrophie)
- `brain_report.json` — Rapport diagnostic

---

## Étape 1 : Créer le Material Heatmap

1. Dans Unity, **Project** → clic droit dans `Assets/` → `Create` → `Material`
2. **Renommer** en `HeatmapMaterial`
3. Sélectionner `HeatmapMaterial`, dans l'Inspector :
   - Cliquer sur le dropdown **Shader** en haut
   - Choisir `BrainXR` → `VertexColor`
4. Glisser `HeatmapMaterial` dans le champ **Heatmap Material** du `MeshImporter`

> Le shader `VertexColor.shader` est dans `Assets/Shaders/`. Il permet d'afficher les couleurs vertex (vert→rouge) du heatmap OBJ.

---

## Étape 2 : Copier les fichiers dans Unity

Copiez les 3 fichiers dans :
```
unity/BrainXR/Assets/StreamingAssets/Models/
```
> Le script Python le fait automatiquement si le dossier Unity existe.

---

## Étape 2 : Créer le panel Diagnostic dans Unity

1. **Ouvrir** votre scène principale dans Unity
2. **Clic droit** dans la Hiérarchie → `UI > Panel`
3. **Renommer** en `PanelDiagnostic`
4. Ajouter le composant **DiagnosticPanel** :
   - Sélectionner `PanelDiagnostic`
   - Inspector → `Add Component` → taper `DiagnosticPanel` → ajouter

---

## Étape 3 : Créer les éléments UI enfants

Dans le `PanelDiagnostic`, créer (clic droit → UI →) :

| N° | Type | Nom | Description |
|----|------|-----|-------------|
| 1 | `Text - TextMeshPro` | `TxtDiagnostic` | Diagnostic principal (ex: "Sain") |
| 2 | `Text - TextMeshPro` | `TxtConfidence` | Pourcentage de confiance |
| 3 | `Image` | `ImgDiagnosticBg` | Fond coloré derrière le diagnostic |
| 4 | `Slider` | `SliderSain` | Barre probabilité "Sain" |
| 5 | `Text - TextMeshPro` | `TxtSain` | Label "Sain : XX%" |
| 6 | `Slider` | `SliderLeger` | Barre probabilité "Déclin léger" |
| 7 | `Text - TextMeshPro` | `TxtLeger` | Label "Déclin léger : XX%" |
| 8 | `Slider` | `SliderAlzheimer` | Barre probabilité "Alzheimer" |
| 9 | `Text - TextMeshPro` | `TxtAlzheimer` | Label "Alzheimer : XX%" |
| 10 | `Text - TextMeshPro` | `TxtAtrophieGlobal` | Atrophie globale |
| 11 | `Text - TextMeshPro` | `TxtRegions` | Liste des régions atrophiées |
| 12 | `Text - TextMeshPro` | `TxtPatientInfo` | Infos patient |
| 13 | `Button - TextMeshPro` | `BtnToggleHeatmap` | Bouton normal ↔ heatmap |

---

## Étape 4 : Connecter les références

1. Sélectionner `PanelDiagnostic` dans la Hiérarchie
2. Dans l'Inspector, section **DiagnosticPanel** :
   - Glisser-déposer chaque élément UI dans le champ correspondant
   - Glisser l'objet **BrainModel** (qui a le `MeshImporter`) dans le champ `Mesh Importer`

---

## Étape 5 : Connecter au UIManager

1. Sélectionner l'objet qui a le composant `UIManager`
2. Dans l'Inspector :
   - Glisser `PanelDiagnostic` dans le champ **Panel Diagnostic**
   - Glisser le composant `DiagnosticPanel` dans le champ **Diagnostic Panel**
3. Optionnel : dans l'écran Visualisation, ajouter un bouton "Diagnostic IA" et le glisser dans le champ **Btn Show Diagnostic**

---

## Étape 6 : Tester

1. Appuyez sur **Play** ▶
2. Naviguez vers l'écran **Diagnostic IA**
3. Vous devriez voir :
   - Le diagnostic coloré (vert/orange/rouge)
   - Les barres de probabilité
   - La liste des régions atrophiées
   - Le bouton pour basculer en heatmap

> **Astuce** : Pour tester avec un autre cerveau, relancez `python download_real_mri.py` et recopiez les fichiers.
