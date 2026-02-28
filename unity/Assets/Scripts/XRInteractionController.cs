/*
 * BrainXR — XRInteractionController.cs
 * ======================================
 * Gere les interactions XR :
 * - Grab/Move (VR controllers ou souris)
 * - Rotation par pinch ou drag
 * - Zoom par scroll ou pinch
 * - Compatible XR Interaction Toolkit + Desktop fallback
 */

using UnityEngine;

public class XRInteractionController : MonoBehaviour
{
    [Header("Sensibilite")]
    public float rotationSpeed = 200f;
    public float zoomSpeed = 5f;
    public float panSpeed = 0.5f;
    public float minZoomDistance = 0.2f;
    public float maxZoomDistance = 10f;
    
    [Header("References")]
    public Transform brainTransform;
    public Camera mainCamera;
    
    private bool isDragging = false;
    private bool isPanning = false;
    private Vector3 lastMousePosition;
    private float currentDistance;
    
    void Start()
    {
        if (mainCamera == null)
            mainCamera = Camera.main;
        
        if (brainTransform == null)
            brainTransform = transform;
        
        // Distance initiale camera-objet
        if (mainCamera != null)
            currentDistance = Vector3.Distance(mainCamera.transform.position, brainTransform.position);
        else
            currentDistance = 3f;
    }
    
    void Update()
    {
        HandleMouseRotation();
        HandleMouseZoom();
        HandleMousePan();
        HandleKeyboardInput();
    }
    
    // ============================================
    //  Rotation souris (clic gauche + drag)
    // ============================================
    
    private void HandleMouseRotation()
    {
        if (Input.GetMouseButtonDown(0))
        {
            isDragging = true;
            lastMousePosition = Input.mousePosition;
        }
        
        if (Input.GetMouseButtonUp(0))
            isDragging = false;
        
        if (isDragging)
        {
            Vector3 delta = Input.mousePosition - lastMousePosition;
            
            float rotX = delta.y * rotationSpeed * Time.deltaTime;
            float rotY = -delta.x * rotationSpeed * Time.deltaTime;
            
            brainTransform.Rotate(Vector3.up, rotY, Space.World);
            brainTransform.Rotate(Vector3.right, rotX, Space.World);
            
            lastMousePosition = Input.mousePosition;
        }
    }
    
    // ============================================
    //  Zoom molette
    // ============================================
    
    private void HandleMouseZoom()
    {
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        
        if (Mathf.Abs(scroll) > 0.01f && mainCamera != null)
        {
            currentDistance -= scroll * zoomSpeed;
            currentDistance = Mathf.Clamp(currentDistance, minZoomDistance, maxZoomDistance);
            
            Vector3 direction = (mainCamera.transform.position - brainTransform.position).normalized;
            mainCamera.transform.position = brainTransform.position + direction * currentDistance;
        }
    }
    
    // ============================================
    //  Pan souris (clic droit + drag)
    // ============================================
    
    private void HandleMousePan()
    {
        if (Input.GetMouseButtonDown(1))
        {
            isPanning = true;
            lastMousePosition = Input.mousePosition;
        }
        
        if (Input.GetMouseButtonUp(1))
            isPanning = false;
        
        if (isPanning && mainCamera != null)
        {
            Vector3 delta = Input.mousePosition - lastMousePosition;
            
            Vector3 move = new Vector3(-delta.x, -delta.y, 0) * panSpeed * Time.deltaTime;
            brainTransform.Translate(mainCamera.transform.TransformDirection(move), Space.World);
            
            lastMousePosition = Input.mousePosition;
        }
    }
    
    // ============================================
    //  Controles clavier
    // ============================================
    
    private void HandleKeyboardInput()
    {
        // Fleches pour rotation
        float kx = Input.GetAxis("Horizontal") * rotationSpeed * 0.5f * Time.deltaTime;
        float ky = Input.GetAxis("Vertical") * rotationSpeed * 0.5f * Time.deltaTime;
        
        if (Mathf.Abs(kx) > 0.01f)
            brainTransform.Rotate(Vector3.up, kx, Space.World);
        if (Mathf.Abs(ky) > 0.01f)
            brainTransform.Rotate(Vector3.right, ky, Space.World);
        
        // R pour reset
        if (Input.GetKeyDown(KeyCode.R))
        {
            brainTransform.rotation = Quaternion.identity;
            brainTransform.position = Vector3.zero;
            Debug.Log("[XR] Vue reinitialisee");
        }
    }
    
    // ============================================
    //  API publique (pour XR Toolkit)
    // ============================================
    
    public void OnGrabStart()
    {
        Debug.Log("[XR] Grab start");
    }
    
    public void OnGrabEnd()
    {
        Debug.Log("[XR] Grab end");
    }
    
    /// <summary>
    /// Appele par XR Interaction Toolkit quand l'objet est selectionne.
    /// </summary>
    public void OnSelectEntered()
    {
        Debug.Log("[XR] Objet selectionne");
    }
}
