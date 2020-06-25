using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

// ===============================
// AUTHOR: Thomas Rood
// PURPOSE: Class that handles the activation of the correct Ball object and keeps track of the last (visual) Ball touch event time.
// SPECIAL NOTES: Note that the environment contains a separate Ball object for each condition, to visually place it in the right position.
// ===============================
public class BallHandler : MonoBehaviour
{
    [Tooltip("Parameters object")]
    public GameObject parameterObject;

    private Parameters parameterScript;
    private GameObject ball_l;
    private GameObject ball_c;
    private GameObject ball_r;
    private float lastTouch;
    private object lastTouchLock = new object();

    /// <summary>
    /// Start is called before the first frame.
    /// The function calls SetActiveBall to activate the correct ball and 
    /// makes sure this function is called upon every reset of the environment in case the parameters changed.
    /// </summary>
    void Start()
    {
        Academy.Instance.OnEnvironmentReset += () =>
        {
            SetActiveBall();
        };

        SetActiveBall();

        lastTouch = -60f;
    }

    /// <summary>
    /// Activates the correct ball object according to the parameter setting.
    /// </summary>
    private void SetActiveBall()
    {
        parameterScript = parameterObject.GetComponent<Parameters>();

        ball_l = this.transform.Find("ball_l").gameObject;
        ball_c = this.transform.Find("ball_c").gameObject;
        ball_r = this.transform.Find("ball_r").gameObject;

        ball_l.SetActive(parameterScript.condition == Parameters.Condition.Left);
        ball_c.SetActive(parameterScript.condition == Parameters.Condition.Center);
        ball_r.SetActive(parameterScript.condition == Parameters.Condition.Right);
    }

    /// <summary>
    /// Called when the Ball reaches the lowest point of the animation.
    /// Sets the last (visual) touch time.
    /// </summary>
    /// <param name="touchTime"></param>
    public void TouchCallBack(float touchTime)
    {
        lock (lastTouchLock)
        {
            lastTouch = touchTime;
        }
    }

    /// <summary>
    /// Get the last (visual) touch event time.
    /// </summary>
    /// <returns>float lastTouch</returns>
    public float GetLastTouch()
    {
        lock (lastTouchLock)
        {
            return lastTouch;
        }
    }

    
}
