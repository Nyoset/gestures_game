using UnityEngine;
using System.Collections.Generic;

public class InputHandler : MonoBehaviour
{
    public delegate void DirectionListener(float value);
    public DirectionListener horizontalAxisListener = delegate (float value) { };
    public DirectionListener verticalAxisListener = delegate (float value) { };

    public delegate void KeyListener();
    public KeyListener enterListener;
    public KeyListener upArrowListener;
    public KeyListener downArrowListener;

    private List<Vector2> gestureRegister = new List<Vector2>();

    void Start()
    {
    }

    void Update()
    {
        horizontalAxisListener(Input.GetAxisRaw("Horizontal"));
        verticalAxisListener(Input.GetAxisRaw("Vertical"));

        if (Input.GetKeyDown(KeyCode.Return)) enterListener?.Invoke();
        if (Input.GetKeyDown(KeyCode.UpArrow)) upArrowListener?.Invoke();
        if (Input.GetKeyDown(KeyCode.DownArrow)) downArrowListener?.Invoke();

        if (Input.touchCount == 1) // user is touching the screen with a single finger
        {
            Touch touch = Input.GetTouch(0); // get the touch
            if (touch.phase == TouchPhase.Began) //check for the first touch
            {
                gestureRegister = new List<Vector2>() { touch.position };
            }
            else if (touch.phase == TouchPhase.Moved) // update the last position based on where they moved
            {
                gestureRegister.Add(touch.position);
            }
            else if (touch.phase == TouchPhase.Ended) //check if the finger is removed from the screen
            {
                for (int i = 0; i < gestureRegister.Count - 1; ++i)
                {
                    Debug.DrawLine(To3D(gestureRegister[i]), To3D(gestureRegister[i+1]), Color.white);
                }
            }
        }
    }

    Vector3 To3D(Vector2 v)
    {
        return new Vector3(v.x, v.y, 0);
    }
}
