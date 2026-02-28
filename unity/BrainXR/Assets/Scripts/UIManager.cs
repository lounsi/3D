/*
 * BrainXR — UIManager.cs
 * =======================
 * Gere l'interface utilisateur a 3 ecrans :
 * 1. Import — charger les images / modele
 * 2. Traitement — lancer le pipeline IA
 * 3. Visualisation — explorer le cerveau en XR
 */

using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using TMPro;

public class UIManager : MonoBehaviour
{
    // ============================================
    //  Panels (3 ecrans)
    // ============================================
    
    [Header("Panels")]
    public GameObject panelImport;
    public GameObject panelTraitement;
    public GameObject panelVisualisation;
    
    // ============================================
    //  Ecran 1 : Import
    // ============================================
    
    [Header("Ecran Import")]
    public Button btnImportImages;
    public Button btnLoadModel;
    public TMP_Text txtSliceCount;
    public TMP_Text txtImportStatus;
    public Image imgPreview;
    public Button btnNextToTraitement;
    
    // ============================================
    //  Ecran 2 : Traitement
    // ============================================
    
    [Header("Ecran Traitement")]
    public Button btnLancerSegmentation;
    public Button btnGenererModele;
    public Slider progressBar;
    public TMP_Text txtProgress;
    public TMP_Text txtLog;
    public Button btnNextToVisu;
    
    // ============================================
    //  Ecran 3 : Visualisation
    // ============================================
    
    [Header("Ecran Visualisation")]
    public Slider sliderTransparence;
    public Slider sliderCoupe;
    public Toggle toggleAutoRotation;
    public Button btnResetView;
    public TMP_Text txtMeshInfo;
    
    // ============================================
    //  References
    // ============================================
    
    [Header("References")]
    public MeshImporter meshImporter;
    public TransparencyController transparencyController;
    public SliceViewer sliceViewer;
    public BrainRotation brainRotation;
    
    // ============================================
    //  Etat
    // ============================================
    
    private int currentScreen = 0;
    private string logContent = "";
    
    void Start()
    {
        // Ecran initial
        ShowScreen(0);
        
        // Boutons navigation
        if (btnNextToTraitement != null)
            btnNextToTraitement.onClick.AddListener(() => ShowScreen(1));
        if (btnNextToVisu != null)
            btnNextToVisu.onClick.AddListener(() => ShowScreen(2));
        
        // Ecran Import
        if (btnLoadModel != null)
            btnLoadModel.onClick.AddListener(OnLoadModel);
        
        // Ecran Traitement
        if (btnGenererModele != null)
            btnGenererModele.onClick.AddListener(OnGenererModele);
        if (progressBar != null)
            progressBar.value = 0;
        
        // Ecran Visualisation
        if (sliderTransparence != null)
            sliderTransparence.onValueChanged.AddListener(OnTransparenceChanged);
        if (sliderCoupe != null)
            sliderCoupe.onValueChanged.AddListener(OnCoupeChanged);
        if (toggleAutoRotation != null)
            toggleAutoRotation.onValueChanged.AddListener(OnAutoRotationChanged);
        if (btnResetView != null)
            btnResetView.onClick.AddListener(OnResetView);
        
        // Callbacks mesh
        if (meshImporter != null)
        {
            meshImporter.OnMeshLoaded += OnMeshLoaded;
            meshImporter.OnMeshError += OnMeshLoadError;
        }
        
        AddLog("Systeme initialise. Pret.");
    }
    
    // ============================================
    //  Navigation entre ecrans
    // ============================================
    
    public void ShowScreen(int index)
    {
        currentScreen = index;
        
        if (panelImport != null)
            panelImport.SetActive(index == 0);
        if (panelTraitement != null)
            panelTraitement.SetActive(index == 1);
        if (panelVisualisation != null)
            panelVisualisation.SetActive(index == 2);
        
        string[] names = { "Import", "Traitement", "Visualisation" };
        Debug.Log($"[UIManager] Ecran actif : {names[index]}");
    }
    
    // ============================================
    //  Actions Ecran Import
    // ============================================
    
    private void OnLoadModel()
    {
        AddLog("Chargement du modele 3D...");
        if (meshImporter != null)
        {
            meshImporter.LoadMesh();
        }
        else
        {
            AddLog("ERREUR : MeshImporter non assigne.");
        }
    }
    
    private void OnMeshLoaded(Mesh mesh)
    {
        AddLog($"Modele charge : {mesh.vertexCount} vertices, {mesh.triangles.Length / 3} faces");
        if (txtMeshInfo != null)
            txtMeshInfo.text = $"Vertices: {mesh.vertexCount}\nFaces: {mesh.triangles.Length / 3}";
        
        UpdateProgress(1.0f, "Modele pret !");
        
        // Passer automatiquement a l'ecran Visualisation
        ShowScreen(2);
    }
    
    private void OnMeshLoadError(string error)
    {
        AddLog($"ERREUR : {error}");
    }
    
    // ============================================
    //  Actions Ecran Traitement
    // ============================================
    
    private void OnGenererModele()
    {
        AddLog("Generation du modele 3D...");
        StartCoroutine(SimulateProcessing());
    }
    
    /// <summary>
    /// Simule la progression du pipeline IA.
    /// En production, cela appellerait le backend Python.
    /// </summary>
    private IEnumerator SimulateProcessing()
    {
        string[] steps = {
            "Pretraitement des images...",
            "Normalisation...",
            "Segmentation en cours...",
            "Reconstruction 3D...",
            "Lissage du volume...",
            "Generation du maillage...",
            "Export OBJ..."
        };
        
        for (int i = 0; i < steps.Length; i++)
        {
            float progress = (float)(i + 1) / steps.Length;
            AddLog(steps[i]);
            UpdateProgress(progress, steps[i]);
            yield return new WaitForSeconds(0.8f);
        }
        
        AddLog("Pipeline termine. Chargement du modele...");
        if (meshImporter != null)
            meshImporter.LoadMesh();
    }
    
    public void UpdateProgress(float value, string message)
    {
        if (progressBar != null)
            progressBar.value = value;
        if (txtProgress != null)
            txtProgress.text = $"{(value * 100):F0}% — {message}";
    }
    
    // ============================================
    //  Actions Ecran Visualisation
    // ============================================
    
    private void OnTransparenceChanged(float value)
    {
        if (transparencyController != null)
            transparencyController.SetTransparency(value);
    }
    
    private void OnCoupeChanged(float value)
    {
        if (sliceViewer != null)
            sliceViewer.SetClipPosition(value);
    }
    
    private void OnAutoRotationChanged(bool enabled)
    {
        if (brainRotation != null)
            brainRotation.SetAutoRotation(enabled);
    }
    
    private void OnResetView()
    {
        if (brainRotation != null)
            brainRotation.ResetRotation();
        if (transparencyController != null)
            transparencyController.SetTransparency(1.0f);
        if (sliceViewer != null)
            sliceViewer.SetClipPosition(1.0f);
        if (sliderTransparence != null)
            sliderTransparence.value = 1.0f;
        if (sliderCoupe != null)
            sliderCoupe.value = 1.0f;
        
        AddLog("Vue reinitialisee.");
    }
    
    // ============================================
    //  Logging
    // ============================================
    
    public void AddLog(string message)
    {
        string timestamp = System.DateTime.Now.ToString("HH:mm:ss");
        logContent += $"[{timestamp}] {message}\n";
        
        if (txtLog != null)
            txtLog.text = logContent;
        
        Debug.Log($"[BrainXR] {message}");
    }
}
