using UnityEngine;

public class MouseRotate : MonoBehaviour
{
    public float speed = 5f;

    void Update()
    {
        if (Input.GetMouseButton(0))
        {
            float h = speed * Input.GetAxis("Mouse X");
            float v = speed * Input.GetAxis("Mouse Y");

            transform.Rotate(Vector3.down, h, Space.World);
            transform.Rotate(Vector3.right, v, Space.World);
        }
    }
}
