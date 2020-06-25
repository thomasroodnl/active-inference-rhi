using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using MLAgents;

// ===============================
// AUTHOR: Thomas Rood
// PURPOSE: Class that generates and keeps track of vibration (tactile) events for every touch event of the ball.
// ===============================
public class VibHandler : MonoBehaviour
{
    [Tooltip("Parameters object")]
    public GameObject parameterObject;

    private Parameters parameterScript;
    private GameObject vib;
    private float lastVib;
    private float newVib;
    private bool vibChanged;
    private float displayTime = 0.1f;

    private float delayLowerBound;
    private float delayUpperBound;

    private System.Random rand = new System.Random();

    // Start is called before the first frame update
    void Start()
    {
        Academy.Instance.OnEnvironmentReset += () =>
        {
            ConfigureVibration();
        };
        ConfigureVibration();
        lastVib = -60f;
        newVib = -60f;
        vibChanged = false;
    }

    /// <summary>
    /// Activates the right vibration object according to the parameter setting.
    /// Configures the vibration (tactile) event delay according to the synchronous/asynchronous parameter setting.
    /// </summary>
    private void ConfigureVibration()
    {
        parameterScript = parameterObject.GetComponent<Parameters>();

        switch (parameterScript.condition)
        {
            case Parameters.Condition.Left:
                vib = this.transform.Find("vib_l").gameObject;
                break;
            case Parameters.Condition.Center:
                vib = this.transform.Find("vib_c").gameObject;
                break;
            case Parameters.Condition.Right:
                vib = this.transform.Find("vib_r").gameObject;
                break;
        }
        switch (parameterScript.stimulation)
        {
            case Parameters.Stimulation.synchronous:
                delayLowerBound = 0.0f;
                delayUpperBound = 0.1f;
                break;
            case Parameters.Stimulation.asynchronous:
                delayLowerBound = 0.0f;
                delayUpperBound = 1f;
                break;
        }
    }

    /// <summary>
    /// Called when the Ball reaches the lowest point of the animation.
    /// Sets the last (tactile) touch time by adding a random delay to the visual touch time.
    /// </summary>
    /// <param name="touchTime"></param>
    public void TouchCallBack(float touchTime)
    {
        newVib = touchTime + delayLowerBound + ((delayUpperBound - delayLowerBound) * (float) rand.NextDouble());
        vibChanged = true;
    }

    /// <summary>
    /// Get the last vibration (tactile) event time.
    /// </summary>
    /// <returns>float lastVib</returns>
    public float GetLastVib()
    {
        if(newVib <= Time.time && vibChanged)
        {
            lastVib = newVib;
            vibChanged = false;
        }
        return lastVib;
    }

    // Update is called once per frame
    void Update()
    {
        // Show the green vibration indicator in the environment when we reach the lastVib time specified by TouchCallBack(float touchTime).
        if (vib != null) { 
            if (Time.time - lastVib < displayTime && !vib.activeSelf)
            {
                vib.SetActive(true);
            } else
            {
                if (Time.time - lastVib >= displayTime && vib.activeSelf)
                {
                    vib.SetActive(false);
                }
            }
        }
    }
}
