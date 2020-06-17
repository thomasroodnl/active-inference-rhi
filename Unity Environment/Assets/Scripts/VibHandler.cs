using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using MLAgents;

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
            Reset();
        };
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
        lastVib = -60f;
        newVib = -60f;
        vibChanged = false;
    }

    private void Reset()
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
                // TODO update delays
                delayLowerBound = 0.0f;
                delayUpperBound = 1f;
                break;
        }
}

    public void TouchCallBack(float touchTime)
    {
        newVib = touchTime + delayLowerBound + ((delayUpperBound - delayLowerBound) * (float) rand.NextDouble());
        vibChanged = true;
    }

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
