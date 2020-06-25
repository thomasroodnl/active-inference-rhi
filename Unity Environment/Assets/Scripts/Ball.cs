using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// ===============================
// AUTHOR: Thomas Rood
// PURPOSE: Intermediate class that propagates the Ball's animation callback to the BallHandler and VibHandler objects.
// ===============================
public class Ball : MonoBehaviour
{
    [Tooltip("BallHandler object")]
    public GameObject ballHandler;

    [Tooltip("VibHandler object")]
    public GameObject vibHandler;

    private BallHandler ballHandlerScript;
    private VibHandler vibHandlerScript;

    // Start is called before the first frame update
    void Start()
    {
        ballHandlerScript = ballHandler.GetComponent<BallHandler>();
        vibHandlerScript = vibHandler.GetComponent<VibHandler>();
    }

    /// <summary>
    /// Function called when the Ball animation is at the lowest point (i.e. touches the hand).
    /// Given that the animation is two seconds long, this function is called every two seconds.
    /// </summary>
    public void TouchCallBack()
    {
        float touchTime = Time.time;
        ballHandlerScript.TouchCallBack(touchTime);
        vibHandlerScript.TouchCallBack(touchTime);
    }
}
