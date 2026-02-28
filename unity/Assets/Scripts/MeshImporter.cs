/*
 * BrainXR — MeshImporter.cs
 * ==========================
 * Charge un fichier OBJ a runtime depuis StreamingAssets 
 * et l'affiche comme MeshFilter/MeshRenderer Unity.
 */

using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Globalization;

public class MeshImporter : MonoBehaviour
{
    [Header("Configuration")]
    [Tooltip("Nom du fichier OBJ dans StreamingAssets/Models/")]
    public string meshFileName = "brain.obj";
    
    [Tooltip("Material a appliquer au mesh normal")]
    public Material brainMaterial;
    
    [Tooltip("Material pour la heatmap (doit supporter les vertex colors)")]
    public Material heatmapMaterial;
    
    [Tooltip("Echelle du modele")]
    public float importScale = 1.0f;
    
    [Header("References")]
    public MeshFilter meshFilter;
    public MeshRenderer meshRenderer;
    public MeshCollider meshCollider;
    
    // Evenements
    public System.Action<Mesh> OnMeshLoaded;
    public System.Action<string> OnMeshError;
    
    private Mesh loadedMesh;
    
    /// <summary>
    /// Charge le mesh OBJ depuis StreamingAssets.
    /// Appelee par le UIManager quand l'utilisateur clique "Charger modele".
    /// </summary>
    public void LoadMesh()
    {
        string path = Path.Combine(Application.streamingAssetsPath, "Models", meshFileName);
        LoadMeshFromPath(path);
    }
    
    /// <summary>
    /// Charge un mesh OBJ depuis un chemin arbitraire.
    /// Detecte automatiquement les vertex colors (heatmap).
    /// </summary>
    public void LoadMeshFromPath(string filePath)
    {
        if (!File.Exists(filePath))
        {
            string error = $"Fichier introuvable : {filePath}";
            Debug.LogError($"[MeshImporter] {error}");
            OnMeshError?.Invoke(error);
            return;
        }
        
        Debug.Log($"[MeshImporter] Chargement de '{filePath}'...");
        
        try
        {
            bool hasVertexColors;
            loadedMesh = ParseOBJ(filePath, out hasVertexColors);
            ApplyMesh(loadedMesh, hasVertexColors);
            Debug.Log($"[MeshImporter] Mesh charge : {loadedMesh.vertexCount} vertices, {loadedMesh.triangles.Length / 3} triangles, colors={hasVertexColors}");
            OnMeshLoaded?.Invoke(loadedMesh);
        }
        catch (System.Exception e)
        {
            string error = $"Erreur lors du chargement : {e.Message}";
            Debug.LogError($"[MeshImporter] {error}");
            OnMeshError?.Invoke(error);
        }
    }
    
    /// <summary>
    /// Parse un fichier OBJ et retourne un Mesh Unity.
    /// Supporte vertices (v), normales (vn), faces (f),
    /// et vertex colors (v x y z r g b) pour les heatmaps.
    /// </summary>
    private Mesh ParseOBJ(string filePath, out bool hasVertexColors)
    {
        List<Vector3> vertices = new List<Vector3>();
        List<Color> colors = new List<Color>();
        List<Vector3> normals = new List<Vector3>();
        List<int> triangles = new List<int>();
        hasVertexColors = false;
        
        string[] lines = File.ReadAllLines(filePath);
        
        foreach (string line in lines)
        {
            string trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#"))
                continue;
            
            string[] parts = trimmed.Split(new char[] { ' ', '\t' }, 
                System.StringSplitOptions.RemoveEmptyEntries);
            
            if (parts.Length == 0) continue;
            
            switch (parts[0])
            {
                case "v":
                    if (parts.Length >= 4)
                    {
                        float x = float.Parse(parts[1], CultureInfo.InvariantCulture);
                        float y = float.Parse(parts[2], CultureInfo.InvariantCulture);
                        float z = float.Parse(parts[3], CultureInfo.InvariantCulture);
                        vertices.Add(new Vector3(x, y, z) * importScale);
                        
                        // Vertex colors (format: v x y z r g b)
                        if (parts.Length >= 7)
                        {
                            float r = float.Parse(parts[4], CultureInfo.InvariantCulture);
                            float g = float.Parse(parts[5], CultureInfo.InvariantCulture);
                            float b = float.Parse(parts[6], CultureInfo.InvariantCulture);
                            colors.Add(new Color(r, g, b, 1.0f));
                            hasVertexColors = true;
                        }
                        else
                        {
                            colors.Add(Color.white);
                        }
                    }
                    break;
                    
                case "vn":
                    if (parts.Length >= 4)
                    {
                        float nx = float.Parse(parts[1], CultureInfo.InvariantCulture);
                        float ny = float.Parse(parts[2], CultureInfo.InvariantCulture);
                        float nz = float.Parse(parts[3], CultureInfo.InvariantCulture);
                        normals.Add(new Vector3(nx, ny, nz));
                    }
                    break;
                    
                case "f":
                    // Supporte les formats : f v, f v//vn, f v/vt/vn
                    List<int> faceIndices = new List<int>();
                    for (int i = 1; i < parts.Length; i++)
                    {
                        string[] indices = parts[i].Split('/');
                        int vertexIndex = int.Parse(indices[0]) - 1; // OBJ est 1-indexed
                        faceIndices.Add(vertexIndex);
                    }
                    // Triangulation (fan triangulation pour polygones)
                    for (int i = 1; i < faceIndices.Count - 1; i++)
                    {
                        triangles.Add(faceIndices[0]);
                        triangles.Add(faceIndices[i]);
                        triangles.Add(faceIndices[i + 1]);
                    }
                    break;
            }
        }
        
        Mesh mesh = new Mesh();
        mesh.name = "BrainModel";
        
        // Support large meshes (>65k vertices)
        if (vertices.Count > 65535)
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        
        // Vertex colors
        if (hasVertexColors && colors.Count == vertices.Count)
            mesh.SetColors(colors);
        
        if (normals.Count == vertices.Count)
            mesh.SetNormals(normals);
        else
            mesh.RecalculateNormals();
        
        mesh.RecalculateBounds();
        mesh.RecalculateTangents();
        
        return mesh;
    }
    
    private void ApplyMesh(Mesh mesh, bool useVertexColors = false)
    {
        if (meshFilter == null)
            meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null)
            meshFilter = gameObject.AddComponent<MeshFilter>();
        
        if (meshRenderer == null)
            meshRenderer = GetComponent<MeshRenderer>();
        if (meshRenderer == null)
            meshRenderer = gameObject.AddComponent<MeshRenderer>();
        
        meshFilter.mesh = mesh;
        
        // Choisir le bon material
        if (useVertexColors && heatmapMaterial != null)
            meshRenderer.material = heatmapMaterial;
        else if (brainMaterial != null)
            meshRenderer.material = brainMaterial;
        
        // Collider pour les interactions XR
        if (meshCollider != null)
        {
            meshCollider.sharedMesh = mesh;
        }
        else
        {
            meshCollider = GetComponent<MeshCollider>();
            if (meshCollider == null)
                meshCollider = gameObject.AddComponent<MeshCollider>();
            meshCollider.sharedMesh = mesh;
        }
    }
    
    /// <summary>
    /// Retourne le mesh charge, ou null.
    /// </summary>
    public Mesh GetLoadedMesh() => loadedMesh;
}
