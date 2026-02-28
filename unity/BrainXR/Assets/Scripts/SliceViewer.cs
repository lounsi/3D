/*
 * BrainXR — SliceViewer.cs
 * =========================
 * Plan de coupe interactif pour visualiser les coupes du cerveau.
 * Utilise un shader avec clipping plane.
 */

using UnityEngine;

public class SliceViewer : MonoBehaviour
{
    [Header("Configuration")]
    [Tooltip("Axe de coupe (0=X, 1=Y, 2=Z)")]
    public int clipAxis = 1; // Y par defaut (coupe axiale)
    
    [Range(0f, 1f)]
    public float clipPosition = 1.0f; // 1.0 = pas de coupe
    
    [Header("References")]
    public MeshRenderer targetRenderer;
    public Transform clipPlaneVisual; // Objet visuel pour le plan de coupe
    
    private Material material;
    private Bounds meshBounds;
    private bool isInitialized = false;
    
    void Start()
    {
        Initialize();
    }
    
    public void Initialize()
    {
        if (targetRenderer != null)
        {
            material = targetRenderer.material;
            
            MeshFilter mf = targetRenderer.GetComponent<MeshFilter>();
            if (mf != null && mf.mesh != null)
            {
                meshBounds = mf.mesh.bounds;
                isInitialized = true;
            }
        }
    }
    
    void Update()
    {
        if (!isInitialized) return;
        UpdateClipPlane();
    }
    
    /// <summary>
    /// Met a jour la position du plan de coupe dans le shader.
    /// </summary>
    private void UpdateClipPlane()
    {
        if (material == null) return;
        
        // Calculer la position du plan dans l'espace objet
        Vector3 planeNormal = Vector3.zero;
        float planeOffset = 0f;
        
        switch (clipAxis)
        {
            case 0: // X
                planeNormal = Vector3.right;
                planeOffset = Mathf.Lerp(meshBounds.min.x, meshBounds.max.x, clipPosition);
                break;
            case 1: // Y
                planeNormal = Vector3.up;
                planeOffset = Mathf.Lerp(meshBounds.min.y, meshBounds.max.y, clipPosition);
                break;
            case 2: // Z
                planeNormal = Vector3.forward;
                planeOffset = Mathf.Lerp(meshBounds.min.z, meshBounds.max.z, clipPosition);
                break;
        }
        
        // Envoyer au shader
        Vector4 clipPlane = new Vector4(planeNormal.x, planeNormal.y, planeNormal.z, -planeOffset);
        material.SetVector("_ClipPlane", clipPlane);
        
        // Mettre a jour le visuel du plan
        if (clipPlaneVisual != null)
        {
            Vector3 pos = targetRenderer.transform.position;
            pos[clipAxis] = planeOffset;
            clipPlaneVisual.position = pos;
        }
    }
    
    // ============================================
    //  API publique
    // ============================================
    
    /// <summary>
    /// Definit la position de coupe (0 = debut, 1 = fin / pas de coupe).
    /// Appele par le slider UI.
    /// </summary>
    public void SetClipPosition(float value)
    {
        clipPosition = Mathf.Clamp01(value);
    }
    
    /// <summary>
    /// Change l'axe de coupe.
    /// </summary>
    public void SetClipAxis(int axis)
    {
        clipAxis = Mathf.Clamp(axis, 0, 2);
        string[] axisNames = { "X (Sagittal)", "Y (Axial)", "Z (Coronal)" };
        Debug.Log($"[SliceViewer] Axe de coupe : {axisNames[axis]}");
    }
    
    /// <summary>
    /// Active/desactive le plan de coupe.
    /// </summary>
    public void SetClipEnabled(bool enabled)
    {
        if (material != null)
        {
            if (enabled)
                material.EnableKeyword("_CLIP_ENABLED");
            else
                material.DisableKeyword("_CLIP_ENABLED");
        }
        
        if (clipPlaneVisual != null)
            clipPlaneVisual.gameObject.SetActive(enabled);
    }
}
