using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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

    public void TouchCallBack()
    {
        float touchTime = Time.time;
        ballHandlerScript.TouchCallBack(touchTime);
        vibHandlerScript.TouchCallBack(touchTime);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
