/*
 * BrainXR — BrainRotation.cs
 * ============================
 * Gere la rotation automatique et manuelle du modele.
 */

using UnityEngine;

public class BrainRotation : MonoBehaviour
{
    [Header("Auto-Rotation")]
    public bool autoRotate = true;
    public float autoRotateSpeed = 20f;
    public Vector3 autoRotateAxis = Vector3.up;
    
    [Header("Rotation manuelle")]
    public float manualRotateSpeed = 300f;
    
    private Quaternion initialRotation;
    private Vector3 initialPosition;
    
    void Start()
    {
        initialRotation = transform.rotation;
        initialPosition = transform.position;
    }
    
    void Update()
    {
        if (autoRotate)
        {
            transform.Rotate(autoRotateAxis, autoRotateSpeed * Time.deltaTime, Space.World);
        }
    }
    
    // ============================================
    //  API publique
    // ============================================
    
    public void SetAutoRotation(bool enabled)
    {
        autoRotate = enabled;
        Debug.Log($"[BrainRotation] Auto-rotation : {(enabled ? "ON" : "OFF")}");
    }
    
    public void SetAutoRotateSpeed(float speed)
    {
        autoRotateSpeed = speed;
    }
    
    public void ResetRotation()
    {
        transform.rotation = initialRotation;
        transform.position = initialPosition;
        Debug.Log("[BrainRotation] Rotation reinitialisee");
    }
    
    public void RotateManual(Vector2 delta)
    {
        float rotX = delta.y * manualRotateSpeed * Time.deltaTime;
        float rotY = -delta.x * manualRotateSpeed * Time.deltaTime;
        
        transform.Rotate(Vector3.up, rotY, Space.World);
        transform.Rotate(Vector3.right, rotX, Space.World);
    }
}
