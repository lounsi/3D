# BrainXR — Planning de Developpement

## Planning sur 4 semaines

### Semaine 1 : Fondations
| Jour | Tache | Livrable |
|------|-------|----------|
| J1-J2 | Setup environnement Python + dependances | `requirements.txt` installe |
| J3-J4 | Module preprocessing complet + tests | `preprocessing.py` valide |
| J5 | Generateur de donnees synthetiques | `generate_test_data.py` |

### Semaine 2 : Pipeline IA
| Jour | Tache | Livrable |
|------|-------|----------|
| J1-J2 | Module segmentation (Otsu + U-Net) | `segmentation.py` |
| J3-J4 | Module reconstruction (Marching Cubes) | `reconstruction.py` |
| J5 | Module export + CLI | `export_mesh.py`, `main.py` |

### Semaine 3 : Unity
| Jour | Tache | Livrable |
|------|-------|----------|
| J1 | Projet Unity + XR Toolkit setup | Projet cree |
| J2 | MeshImporter + shader | OBJ charge dans Unity |
| J3 | UI 3 ecrans | Interface fonctionnelle |
| J4 | Interactions XR (rotation, zoom) | Navigation OK |
| J5 | SliceViewer + transparence | Coupe interactive |

### Semaine 4 : Integration + Polish
| Jour | Tache | Livrable |
|------|-------|----------|
| J1-J2 | Pipeline end-to-end test | Demo fonctionnelle |
| J3 | Documentation technique | `docs/` complet |
| J4 | Tests sur Meta Quest / AR | Build XR |
| J5 | Presentation + rapport | Soutenance prete |

---

## Risques techniques

| Risque | Impact | Probabilite | Mitigation |
|--------|--------|-------------|------------|
| Pas de GPU disponible | Segmentation lente | Moyen | Fallback Otsu (CPU) |
| Mesh trop lourd pour XR | FPS bas | Moyen | Simplification mesh (50%) |
| Format DICOM complexe | Import echoue | Faible | Support PNG comme alternative |
| Unity XR Toolkit bugs | Interactions KO | Faible | Fallback souris/clavier |
| Qualite segmentation faible | Mesh irregulier | Moyen | Lissage gaussien 3D |

---

## Dependances externes

### Python
```
numpy, opencv-python, scikit-image, SimpleITK,
torch, torchvision, nibabel, trimesh, Pillow, tqdm
```

### Unity
- Unity 2022 LTS
- XR Interaction Toolkit (via Package Manager)
- XR Plugin Management
- TextMeshPro (UI)

### Hardware recommande
- GPU NVIDIA (optionnel, pour U-Net)
- Meta Quest 2/3 (pour VR) ou smartphone Android (pour AR)
- 8 Go RAM minimum
