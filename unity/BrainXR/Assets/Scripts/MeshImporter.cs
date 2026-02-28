using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Globalization;

public class MeshImporter : MonoBehaviour
{
    [Header("Configuration")]
    public string meshFileName = "brain.obj";
    public Material brainMaterial;
    public float importScale = 1f;

    [Header("References")]
    public MeshFilter meshFilter;
    public MeshRenderer meshRenderer;
    public MeshCollider meshCollider;

    // 🔥 Events utilisés par UIManager
    public System.Action<Mesh> OnMeshLoaded;
    public System.Action<string> OnMeshError;

    private Mesh loadedMesh;

    void Start()
    {
        LoadMesh();
    }

    public void LoadMesh()
    {
        string path = Path.Combine(Application.streamingAssetsPath, "Models", meshFileName);

        if (!File.Exists(path))
        {
            string error = "File not found: " + path;
            Debug.LogError(error);
            OnMeshError?.Invoke(error);
            return;
        }

        try
        {
            loadedMesh = ParseOBJ(path);
            ApplyMesh(loadedMesh);

            Debug.Log("Mesh loaded: " + loadedMesh.vertexCount + " vertices");
            OnMeshLoaded?.Invoke(loadedMesh);
        }
        catch (System.Exception e)
        {
            string error = "Error loading mesh: " + e.Message;
            Debug.LogError(error);
            OnMeshError?.Invoke(error);
        }
    }

    private Mesh ParseOBJ(string filePath)
    {
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();

        string[] lines = File.ReadAllLines(filePath);

        foreach (string line in lines)
        {
            if (line.StartsWith("v "))
            {
                string[] p = line.Split(' ');
                float x = float.Parse(p[1], CultureInfo.InvariantCulture);
                float y = float.Parse(p[2], CultureInfo.InvariantCulture);
                float z = float.Parse(p[3], CultureInfo.InvariantCulture);
                vertices.Add(new Vector3(x, y, z) * importScale);
            }

            if (line.StartsWith("f "))
            {
                string[] p = line.Split(' ');
                triangles.Add(int.Parse(p[1].Split('/')[0]) - 1);
                triangles.Add(int.Parse(p[2].Split('/')[0]) - 1);
                triangles.Add(int.Parse(p[3].Split('/')[0]) - 1);
            }
        }

        Mesh mesh = new Mesh();

        if (vertices.Count > 65535)
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        return mesh;
    }

    private void ApplyMesh(Mesh mesh)
    {
        if (meshFilter == null)
            meshFilter = GetComponent<MeshFilter>();

        if (meshRenderer == null)
            meshRenderer = GetComponent<MeshRenderer>();

        if (meshCollider == null)
            meshCollider = GetComponent<MeshCollider>();

        meshFilter.mesh = mesh;

        if (brainMaterial != null)
            meshRenderer.material = brainMaterial;

        if (meshCollider != null)
            meshCollider.sharedMesh = mesh;
    }

    public Mesh GetLoadedMesh()
    {
        return loadedMesh;
    }
}
