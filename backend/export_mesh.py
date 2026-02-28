"""
BrainXR - Module d'Export Mesh
===============================
Export OBJ / PLY avec validation.
"""

import os
import numpy as np
from datetime import datetime


def compute_vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1, edge2 = v1 - v0, v2 - v0
        fn = np.cross(edge1, edge2)
        normals[face[0]] += fn
        normals[face[1]] += fn
        normals[face[2]] += fn
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return normals / norms


def export_obj(vertices, faces, output_path, normals=None, comment=None):
    if not output_path.lower().endswith('.obj'):
        output_path += '.obj'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if normals is None:
        normals = compute_vertex_normals(vertices, faces)
    with open(output_path, 'w') as f:
        f.write(f"# BrainXR - Modele 3D du cerveau\n")
        f.write(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}\n")
        if comment:
            f.write(f"# {comment}\n")
        f.write("o BrainModel\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n")
        for face in faces:
            i0, i1, i2 = face[0]+1, face[1]+1, face[2]+1
            f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
    sz = os.path.getsize(output_path) / (1024*1024)
    print(f"  -> OBJ exporte: '{output_path}' ({sz:.2f} MB)")
    return output_path


def export_ply(vertices, faces, output_path, normals=None):
    if not output_path.lower().endswith('.ply'):
        output_path += '.ply'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if normals is None:
        normals = compute_vertex_normals(vertices, faces)
    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"comment BrainXR\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v, n in zip(vertices, normals):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    sz = os.path.getsize(output_path) / (1024*1024)
    print(f"  -> PLY exporte: '{output_path}' ({sz:.2f} MB)")
    return output_path


def validate_mesh(vertices, faces):
    results = {"valid": True, "num_vertices": vertices.shape[0], "num_faces": faces.shape[0], "issues": []}
    if faces.max() >= vertices.shape[0]:
        results["valid"] = False
        results["issues"].append("Index de face hors limites")
    if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
        results["valid"] = False
        results["issues"].append("Vertices contiennent NaN ou Inf")
    return results


def export_pipeline(vertices, faces, output_path, format="obj", normals=None):
    print("\n" + "="*50)
    print("ETAPE 4 - EXPORT")
    print("="*50)
    validation = validate_mesh(vertices, faces)
    if not validation["valid"]:
        raise ValueError(f"Mesh invalide: {validation['issues']}")
    print(f"  Mesh valide ({validation['num_vertices']} v, {validation['num_faces']} f)")
    if format.lower() == "obj":
        return export_obj(vertices, faces, output_path, normals)
    elif format.lower() == "ply":
        return export_ply(vertices, faces, output_path, normals)
    else:
        raise ValueError(f"Format non supporte: '{format}'")
