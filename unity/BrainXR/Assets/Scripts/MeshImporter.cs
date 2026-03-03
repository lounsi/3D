using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Globalization;

public class MeshImporter : MonoBehaviour
{
    [Header("Configuration")]
    public string meshFileName = "brain_heatmap.obj"; 
    public Material brainMaterial;
    public Material heatmapMaterial; 
    public float importScale = 1.0f;

    [Header("References")]
    public MeshFilter meshFilter;
    public MeshRenderer meshRenderer;
    public MeshCollider meshCollider;

    // Evenements pour notifier les autres scripts (UIManager, etc.)
    public event System.Action<Mesh> OnMeshLoaded;
    public event System.Action<string> OnMeshError;

    void Start() 
    { 
        LoadMesh(); 
    }

    public void LoadMesh()
    {
        string path = Path.Combine(Application.streamingAssetsPath, "Models", meshFileName);
        LoadMeshFromPath(path);
    }

    public void LoadMeshFromPath(string filePath)
    {
        if (!File.Exists(filePath))
        {
            string err = "Fichier introuvable : " + filePath;
            Debug.LogError("[MeshImporter] " + err);
            OnMeshError?.Invoke(err);
            return;
        }

        try
        {
            bool hasVertexColors;
            Mesh loadedMesh = ParseOBJ(filePath, out hasVertexColors);
            ApplyMesh(loadedMesh, hasVertexColors);
            Debug.Log("[MeshImporter] Succes : " + loadedMesh.vertexCount + " sommets charges. Couleurs : " + hasVertexColors);
            OnMeshLoaded?.Invoke(loadedMesh);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[MeshImporter] Erreur : " + e.Message);
            OnMeshError?.Invoke(e.Message);
        }
    }

    private Mesh ParseOBJ(string filePath, out bool hasVertexColors)
    {
        List<Vector3> vertices = new List<Vector3>();
        List<Color> colors = new List<Color>();
        List<int> triangles = new List<int>();
        hasVertexColors = false;

        string[] lines = File.ReadAllLines(filePath);
        foreach (string line in lines)
        {
            string trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#")) continue;

            string[] parts = trimmed.Split(new char[] { ' ', '\t' }, System.StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 1) continue;

            if (parts[0] == "v" && parts.Length >= 4)
            {
                float x = float.Parse(parts[1], CultureInfo.InvariantCulture);
                float y = float.Parse(parts[2], CultureInfo.InvariantCulture);
                float z = float.Parse(parts[3], CultureInfo.InvariantCulture);
                vertices.Add(new Vector3(x, y, z) * importScale);

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
            else if (parts[0] == "f")
            {
                for (int i = 1; i < parts.Length - 2; i++)
                {
                    triangles.Add(int.Parse(parts[1].Split('/')[0]) - 1);
                    triangles.Add(int.Parse(parts[i+1].Split('/')[0]) - 1);
                    triangles.Add(int.Parse(parts[i+2].Split('/')[0]) - 1);
                }
            }
        }

        Mesh mesh = new Mesh();
        if (vertices.Count > 65000) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        if (hasVertexColors) mesh.SetColors(colors);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    private void ApplyMesh(Mesh mesh, bool useVertexColors)
    {
        if (meshFilter != null) meshFilter.mesh = mesh;
        Material targetMat = (useVertexColors && heatmapMaterial != null) ? heatmapMaterial : brainMaterial;
        if (meshRenderer != null) meshRenderer.material = targetMat;
        if (meshCollider != null) meshCollider.sharedMesh = mesh;
    }
}