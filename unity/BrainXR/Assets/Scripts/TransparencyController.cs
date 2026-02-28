/*
 * BrainXR — TransparencyController.cs
 * =====================================
 * Controle la transparence du modele 3D du cerveau.
 * Permet de voir a travers le modele pour observer 
 * les structures internes.
 */

using UnityEngine;

public class TransparencyController : MonoBehaviour
{
    [Header("Configuration")]
    [Range(0f, 1f)]
    public float transparency = 1.0f; // 1.0 = opaque
    
    [Header("Couleurs zones segmentees")]
    public Color brainColor = new Color(0.4f, 0.6f, 0.9f, 1f);
    public Color highlightColor = new Color(0.5f, 0.7f, 1f, 1f);
    
    [Header("References")]
    public MeshRenderer targetRenderer;
    
    private Material material;
    private bool isHighlighting = false;
    
    void Start()
    {
        if (targetRenderer != null)
        {
            // Creer une instance du material
            material = new Material(targetRenderer.material);
            targetRenderer.material = material;
            SetTransparency(transparency);
        }
    }
    
    /// <summary>
    /// Definit le niveau de transparence.
    /// 0 = completement transparent, 1 = opaque.
    /// </summary>
    public void SetTransparency(float value)
    {
        transparency = Mathf.Clamp01(value);
        
        if (material == null) return;
        
        Color color = material.color;
        color.a = transparency;
        material.color = color;
        
        // Basculer le mode de rendu
        if (transparency < 0.99f)
        {
            // Mode transparent
            material.SetFloat("_Mode", 3); // Transparent
            material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            material.SetInt("_ZWrite", 0);
            material.DisableKeyword("_ALPHATEST_ON");
            material.EnableKeyword("_ALPHABLEND_ON");
            material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            material.renderQueue = 3000;
        }
        else
        {
            // Mode opaque
            material.SetFloat("_Mode", 0); // Opaque
            material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
            material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.Zero);
            material.SetInt("_ZWrite", 1);
            material.DisableKeyword("_ALPHATEST_ON");
            material.DisableKeyword("_ALPHABLEND_ON");
            material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            material.renderQueue = -1;
        }
    }
    
    /// <summary>
    /// Active/desactive la surbrillance d'une zone.
    /// </summary>
    public void SetHighlight(bool enabled)
    {
        isHighlighting = enabled;
        if (material != null)
        {
            material.color = enabled ? 
                new Color(highlightColor.r, highlightColor.g, highlightColor.b, transparency) :
                new Color(brainColor.r, brainColor.g, brainColor.b, transparency);
        }
    }
    
    /// <summary>
    /// Definit la couleur du cerveau.
    /// </summary>
    public void SetBrainColor(Color color)
    {
        brainColor = color;
        if (!isHighlighting && material != null)
        {
            color.a = transparency;
            material.color = color;
        }
    }
}
