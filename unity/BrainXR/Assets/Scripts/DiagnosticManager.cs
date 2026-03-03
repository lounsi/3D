using UnityEngine;
using TMPro;
using System.IO;
using System;

// ============================================
//  Classes de donnees — correspondent exactement
//  a la structure de brain_report.json
// ============================================

[Serializable]
public class DemographicData
{
    public float age;
    public string sex;
    public float education;
    public float socioeconomic;
    public float mmse;
    public float cdr;
}

[Serializable]
public class ProbabilitiesData
{
    public float Sain;
    // JsonUtility ne supporte pas les espaces dans les cles,
    // on les recupere manuellement dans le parsing
    [NonSerialized] public float DeclinLeger;
    public float Alzheimer;
}

[Serializable]
public class ClassificationData
{
    public string prediction;
    public float confidence;
    public ProbabilitiesData probabilities;
}

[Serializable]
public class RegionData
{
    public string name;
    public float z_score;
    public string status;
    public float deviation_percent;
}

[Serializable]
public class StatusCountsData
{
    public int normal;
    public int mild;
    public int moderate;
    public int severe;
}

[Serializable]
public class AtrophyData
{
    public float global_atrophy_percent;
    public float total_gray_matter_volume;
    public float total_z_score;
    public RegionData[] regions;
    public string[] most_atrophied;
    public StatusCountsData status_counts;
    public int reference_subjects;
}

[Serializable]
public class BrainReport
{
    public int subject_id;
    public string timestamp;
    public DemographicData demographic;
    public ClassificationData classification;
    public AtrophyData atrophy;
    public string mesh_file;
    public string heatmap_mesh_file;
}

// ============================================
//  DiagnosticManager — Affiche le rapport IA
// ============================================

public class DiagnosticManager : MonoBehaviour
{
    [Header("UI References")]
    public TextMeshProUGUI reportText;

    [Header("Configuration")]
    public string fileName = "brain_report.json";

    void Start()
    {
        LoadAndDisplay();
    }

    public void LoadAndDisplay()
    {
        string path = Path.Combine(Application.streamingAssetsPath, "Models", fileName);

        if (!File.Exists(path))
        {
            if (reportText != null)
                reportText.text = "<color=orange>En attente du rapport medical...\nFichier introuvable.</color>";
            Debug.LogWarning("[DiagnosticManager] Fichier non trouve : " + path);
            return;
        }

        try
        {
            string json = File.ReadAllText(path);

            // Recuperer "Declin leger" manuellement (JsonUtility ne gere pas les cles avec espaces)
            float declinLeger = 0f;
            int idx = json.IndexOf("\"Declin leger\"");
            if (idx >= 0)
            {
                int colon = json.IndexOf(':', idx);
                int end = json.IndexOf(',', colon);
                if (end < 0) end = json.IndexOf('}', colon);
                string val = json.Substring(colon + 1, end - colon - 1).Trim();
                float.TryParse(val, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out declinLeger);
            }

            BrainReport data = JsonUtility.FromJson<BrainReport>(json);
            if (data.classification.probabilities != null)
                data.classification.probabilities.DeclinLeger = declinLeger;

            FormatAndDisplay(data);
            Debug.Log("[DiagnosticManager] Rapport charge avec succes.");
        }
        catch (Exception e)
        {
            if (reportText != null)
                reportText.text = "<color=red>Erreur lors de la lecture du rapport.</color>";
            Debug.LogError("[DiagnosticManager] Erreur parsing JSON : " + e.Message);
        }
    }

    void FormatAndDisplay(BrainReport data)
    {
        string r = "";

        // === EN-TETE ===
        r += "<size=130%><b>BILAN DE SANTE CEREBRALE</b></size>\n";
        r += "<color=#888888>Sujet #" + data.subject_id + " — " + data.timestamp + "</color>\n\n";

        // === PATIENT ===
        r += "<size=110%><b>Patient</b></size>\n";
        r += "  Age : " + data.demographic.age + " ans\n";
        r += "  Sexe : " + (data.demographic.sex == "F" ? "Femme" : "Homme") + "\n";
        r += "  Education : " + data.demographic.education + " ans\n";
        r += "  MMSE : " + data.demographic.mmse + "/30\n";
        r += "  CDR : " + data.demographic.cdr + "\n\n";

        // === DIAGNOSTIC IA ===
        r += "<size=110%><b>Diagnostic IA</b></size>\n";
        string diagColor = GetDiagnosticColor(data.classification.prediction);
        r += "  Prediction : <color=" + diagColor + "><b>" + data.classification.prediction + "</b></color>\n";
        r += "  Confiance : " + data.classification.confidence.ToString("F1") + "%\n";

        if (data.classification.probabilities != null)
        {
            r += "  Probabilites :\n";
            r += "    - Sain : " + data.classification.probabilities.Sain.ToString("F1") + "%\n";
            r += "    - Declin leger : " + data.classification.probabilities.DeclinLeger.ToString("F1") + "%\n";
            r += "    - Alzheimer : " + data.classification.probabilities.Alzheimer.ToString("F1") + "%\n";
        }
        r += "\n";

        // === ATROPHIE ===
        r += "<size=110%><b>Atrophie cerebrale</b></size>\n";
        r += "  Atrophie globale : <b>" + data.atrophy.global_atrophy_percent.ToString("F1") + "%</b>\n";
        r += "  Volume matiere grise : " + data.atrophy.total_gray_matter_volume.ToString("F0") + " mm3\n";
        r += "  Z-score global : " + data.atrophy.total_z_score.ToString("F2") + "\n\n";

        // Repartition des regions
        if (data.atrophy.status_counts != null)
        {
            StatusCountsData sc = data.atrophy.status_counts;
            r += "  Repartition (" + data.atrophy.reference_subjects + " sujets ref.) :\n";
            r += "    <color=#55FF55>" + sc.normal + " normales</color>  |  ";
            r += "<color=#FFFF55>" + sc.mild + " legeres</color>  |  ";
            r += "<color=#FF8855>" + sc.moderate + " moderees</color>  |  ";
            r += "<color=#FF5555>" + sc.severe + " severes</color>\n\n";
        }

        // === REGIONS LES PLUS TOUCHEES ===
        if (data.atrophy.regions != null && data.atrophy.regions.Length > 0)
        {
            r += "<size=110%><b>Regions les plus touchees</b></size>\n";
            foreach (RegionData region in data.atrophy.regions)
            {
                string statusColor = GetStatusColor(region.status);
                r += "  <color=" + statusColor + ">\u25CF</color> <b>" + region.name + "</b>\n";
                r += "    Z-score : " + region.z_score.ToString("F2");
                r += "  |  Deviation : " + region.deviation_percent.ToString("F1") + "%";
                r += "  |  " + TranslateStatus(region.status) + "\n";
            }
        }

        reportText.text = r;
    }

    // Couleur du diagnostic selon la gravite
    string GetDiagnosticColor(string prediction)
    {
        switch (prediction)
        {
            case "Sain": return "#55FF55";
            case "Declin leger": return "#FFAA55";
            case "Alzheimer": return "#FF5555";
            default: return "#FFFFFF";
        }
    }

    // Couleur du statut d'atrophie
    string GetStatusColor(string status)
    {
        switch (status)
        {
            case "normal": return "#55FF55";
            case "mild": return "#FFFF55";
            case "moderate": return "#FF8855";
            case "severe": return "#FF5555";
            default: return "#FFFFFF";
        }
    }

    // Traduction du statut en francais
    string TranslateStatus(string status)
    {
        switch (status)
        {
            case "normal": return "Normal";
            case "mild": return "Leger";
            case "moderate": return "Modere";
            case "severe": return "Severe";
            default: return status;
        }
    }
}
