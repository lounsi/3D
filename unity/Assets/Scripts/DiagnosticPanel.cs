/*
 * BrainXR — DiagnosticPanel.cs
 * ==============================
 * Affiche les resultats de l'analyse IA :
 * - Diagnostic Alzheimer (Sain / Declin leger / Alzheimer)
 * - Barres de probabilite
 * - Analyse d'atrophie par region
 * - Bouton pour basculer mesh normal <-> heatmap
 *
 * Lit le fichier brain_report.json depuis StreamingAssets/Models/
 */

using UnityEngine;
using UnityEngine.UI;
using System.IO;
using TMPro;

public class DiagnosticPanel : MonoBehaviour
{
    // ============================================
    //  UI Elements
    // ============================================
    
    [Header("Diagnostic")]
    [Tooltip("Texte principal du diagnostic (ex: 'Sain')")]
    public TMP_Text txtDiagnostic;
    
    [Tooltip("Texte de confiance (ex: '85.2%')")]
    public TMP_Text txtConfidence;
    
    [Tooltip("Image de fond du diagnostic (pour la couleur)")]
    public Image imgDiagnosticBg;
    
    [Header("Probabilites")]
    [Tooltip("Slider probabilite 'Sain'")]
    public Slider sliderSain;
    public TMP_Text txtSain;
    
    [Tooltip("Slider probabilite 'Declin leger'")]
    public Slider sliderLeger;
    public TMP_Text txtLeger;
    
    [Tooltip("Slider probabilite 'Alzheimer'")]
    public Slider sliderAlzheimer;
    public TMP_Text txtAlzheimer;
    
    [Header("Atrophie")]
    [Tooltip("Texte atrophie globale")]
    public TMP_Text txtAtrophieGlobal;
    
    [Tooltip("Texte des regions atrophiees")]
    public TMP_Text txtRegions;
    
    [Header("Patient")]
    [Tooltip("Texte infos patient")]
    public TMP_Text txtPatientInfo;
    
    [Header("Heatmap")]
    [Tooltip("Bouton pour basculer normal <-> heatmap")]
    public Button btnToggleHeatmap;
    public TMP_Text txtToggleHeatmap;
    
    [Header("References")]
    public MeshImporter meshImporter;
    
    // ============================================
    //  Etat interne
    // ============================================
    
    private BrainReport report;
    private bool showingHeatmap = false;
    
    // ============================================
    //  Classe de donnees JSON
    // ============================================
    
    [System.Serializable]
    public class BrainReport
    {
        public int subject_id;
        public string timestamp;
        public Demographic demographic;
        public Classification classification;
        public Atrophy atrophy;
        public string mesh_file;
        public string heatmap_mesh_file;
    }
    
    [System.Serializable]
    public class Demographic
    {
        public float age;
        public float mmse;
        public float cdr;
    }
    
    [System.Serializable]
    public class Classification
    {
        public string prediction;
        public float confidence;
        public Probabilities probabilities;
    }
    
    [System.Serializable]
    public class Probabilities
    {
        public float Sain;
        // JsonUtility ne supporte pas les espaces dans les cles,
        // on parse manuellement si necessaire
        public float Alzheimer;
    }
    
    [System.Serializable]
    public class Atrophy
    {
        public float global_atrophy_percent;
        public float total_gray_matter_volume;
        public float total_z_score;
        public Region[] regions;
        public string[] most_atrophied;
    }
    
    [System.Serializable]
    public class Region
    {
        public string name;
        public float z_score;
        public string status;
        public float deviation_percent;
    }
    
    // ============================================
    //  Initialisation
    // ============================================
    
    void Start()
    {
        if (btnToggleHeatmap != null)
            btnToggleHeatmap.onClick.AddListener(ToggleHeatmap);
        
        LoadReport();
    }
    
    /// <summary>
    /// Charge et parse le rapport JSON.
    /// </summary>
    public void LoadReport()
    {
        string path = Path.Combine(
            Application.streamingAssetsPath, "Models", "brain_report.json"
        );
        
        if (!File.Exists(path))
        {
            Debug.LogWarning("[DiagnosticPanel] brain_report.json introuvable.");
            SetDefaultUI();
            return;
        }
        
        try
        {
            string json = File.ReadAllText(path);
            report = JsonUtility.FromJson<BrainReport>(json);
            
            // Parse manuelle des probabilites (JsonUtility ne gere pas bien les cles avec espaces)
            ParseProbabilities(json);
            
            UpdateUI();
            Debug.Log("[DiagnosticPanel] Rapport charge avec succes.");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[DiagnosticPanel] Erreur de lecture : {e.Message}");
            SetDefaultUI();
        }
    }
    
    // ============================================
    //  Parse manuelle des probabilites
    // ============================================
    
    private float probSain = 0;
    private float probLeger = 0;
    private float probAlzheimer = 0;
    
    private void ParseProbabilities(string json)
    {
        // Extraire les probabilites du JSON brut
        // car JsonUtility ne supporte pas les cles avec accents/espaces
        probSain = ExtractFloatFromJson(json, "Sain");
        probLeger = ExtractFloatFromJson(json, "clin l");
        if (probLeger == 0)
            probLeger = ExtractFloatFromJson(json, "Declin leger");
        probAlzheimer = ExtractFloatFromJson(json, "Alzheimer");
    }
    
    private float ExtractFloatFromJson(string json, string key)
    {
        // Recherche simple : "key": value
        int idx = json.IndexOf($"\"{key}\"");
        if (idx < 0)
        {
            // Essayer sans accent
            key = key.Replace("é", "e").Replace("è", "e");
            idx = json.IndexOf($"\"{key}\"");
        }
        if (idx < 0) return 0;
        
        int colonIdx = json.IndexOf(':', idx);
        if (colonIdx < 0) return 0;
        
        // Trouver la valeur numerique apres le :
        string afterColon = json.Substring(colonIdx + 1, 20).Trim();
        string numStr = "";
        foreach (char c in afterColon)
        {
            if (char.IsDigit(c) || c == '.' || c == '-')
                numStr += c;
            else if (numStr.Length > 0)
                break;
        }
        
        if (float.TryParse(numStr, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out float val))
            return val;
        
        return 0;
    }
    
    // ============================================
    //  Mise a jour de l'UI
    // ============================================
    
    private void UpdateUI()
    {
        if (report == null) return;
        
        // --- Diagnostic principal ---
        if (txtDiagnostic != null)
            txtDiagnostic.text = report.classification.prediction;
        
        if (txtConfidence != null)
            txtConfidence.text = $"{report.classification.confidence:F1}%";
        
        // Couleur selon le diagnostic
        if (imgDiagnosticBg != null)
        {
            switch (report.classification.prediction)
            {
                case "Sain":
                    imgDiagnosticBg.color = new Color(0.2f, 0.75f, 0.3f, 0.8f); // Vert
                    break;
                case "Declin leger":
                    imgDiagnosticBg.color = new Color(1.0f, 0.7f, 0.1f, 0.8f); // Orange
                    break;
                case "Alzheimer":
                    imgDiagnosticBg.color = new Color(0.9f, 0.2f, 0.15f, 0.8f); // Rouge
                    break;
                default:
                    imgDiagnosticBg.color = new Color(0.5f, 0.5f, 0.5f, 0.8f); // Gris
                    break;
            }
        }
        
        // --- Probabilites ---
        SetSlider(sliderSain, txtSain, "Sain", probSain);
        SetSlider(sliderLeger, txtLeger, "Declin leger", probLeger);
        SetSlider(sliderAlzheimer, txtAlzheimer, "Alzheimer", probAlzheimer);
        
        // --- Atrophie globale ---
        if (txtAtrophieGlobal != null)
        {
            float atr = report.atrophy.global_atrophy_percent;
            string severity = atr < 5 ? "Normale" : atr < 15 ? "Legere" : atr < 25 ? "Moderee" : "Severe";
            txtAtrophieGlobal.text = $"Atrophie globale : {atr:F1}% ({severity})";
        }
        
        // --- Regions atrophiees ---
        if (txtRegions != null && report.atrophy.regions != null)
        {
            string text = "<b>Regions analysees :</b>\n";
            int count = Mathf.Min(report.atrophy.regions.Length, 8);
            
            for (int i = 0; i < count; i++)
            {
                Region r = report.atrophy.regions[i];
                string color = GetStatusColor(r.status);
                string icon = GetStatusIcon(r.status);
                text += $"  {icon} <color={color}>{r.name}</color>  " +
                        $"z={r.z_score:F1}  ({r.deviation_percent:F0}%)\n";
            }
            
            txtRegions.text = text;
        }
        
        // --- Infos patient ---
        if (txtPatientInfo != null && report.demographic != null)
        {
            string info = $"<b>Sujet OASIS #{report.subject_id}</b>\n";
            if (report.demographic.age > 0)
                info += $"Age : {report.demographic.age:F0} ans\n";
            if (report.demographic.mmse > 0)
                info += $"MMSE : {report.demographic.mmse:F0}/30\n";
            if (report.demographic.cdr >= 0)
                info += $"CDR reel : {report.demographic.cdr:F1}\n";
            txtPatientInfo.text = info;
        }
        
        // --- Bouton heatmap ---
        UpdateHeatmapButton();
    }
    
    private void SetSlider(Slider slider, TMP_Text label, string name, float value)
    {
        if (slider != null)
        {
            slider.minValue = 0;
            slider.maxValue = 100;
            slider.value = value;
            slider.interactable = false; // Lecture seule
        }
        if (label != null)
            label.text = $"{name} : {value:F1}%";
    }
    
    private string GetStatusColor(string status)
    {
        switch (status)
        {
            case "normal": return "#4CAF50";
            case "mild": return "#FFC107";
            case "moderate": return "#FF9800";
            case "severe": return "#F44336";
            default: return "#9E9E9E";
        }
    }
    
    private string GetStatusIcon(string status)
    {
        switch (status)
        {
            case "normal": return "●";
            case "mild": return "▲";
            case "moderate": return "◆";
            case "severe": return "■";
            default: return "○";
        }
    }
    
    private void SetDefaultUI()
    {
        if (txtDiagnostic != null)
            txtDiagnostic.text = "Aucun rapport";
        if (txtConfidence != null)
            txtConfidence.text = "--";
        if (txtAtrophieGlobal != null)
            txtAtrophieGlobal.text = "Lancez d'abord le pipeline Python";
        if (txtRegions != null)
            txtRegions.text = "";
        if (txtPatientInfo != null)
            txtPatientInfo.text = "";
    }
    
    // ============================================
    //  Toggle Heatmap
    // ============================================
    
    private void ToggleHeatmap()
    {
        showingHeatmap = !showingHeatmap;
        
        if (meshImporter != null)
        {
            string filename = showingHeatmap ? "brain_heatmap.obj" : "brain.obj";
            string path = Path.Combine(
                Application.streamingAssetsPath, "Models", filename
            );
            
            if (File.Exists(path))
            {
                meshImporter.LoadMeshFromPath(path);
                Debug.Log($"[DiagnosticPanel] Mesh bascule vers : {filename}");
            }
            else
            {
                Debug.LogWarning($"[DiagnosticPanel] {filename} introuvable");
                showingHeatmap = !showingHeatmap; // Annuler
            }
        }
        
        UpdateHeatmapButton();
    }
    
    private void UpdateHeatmapButton()
    {
        if (txtToggleHeatmap != null)
        {
            txtToggleHeatmap.text = showingHeatmap
                ? "Voir Normal"
                : "Voir Heatmap Atrophie";
        }
    }
}
