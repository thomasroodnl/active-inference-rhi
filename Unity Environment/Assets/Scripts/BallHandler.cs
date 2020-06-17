using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

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

    // Start is called before the first frame update
    void Start()
    {
        Academy.Instance.OnEnvironmentReset += () =>
        {
            Reset();
        };
        parameterScript = parameterObject.GetComponent<Parameters>();

        ball_l = this.transform.Find("ball_l").gameObject;
        ball_c = this.transform.Find("ball_c").gameObject;
        ball_r = this.transform.Find("ball_r").gameObject;

        ball_l.SetActive(parameterScript.condition == Parameters.Condition.Left);
        ball_c.SetActive(parameterScript.condition == Parameters.Condition.Center);
        ball_r.SetActive(parameterScript.condition == Parameters.Condition.Right);
        lastTouch = -60f;
    }

    private void Reset()
    {
        parameterScript = parameterObject.GetComponent<Parameters>();

        ball_l = this.transform.Find("ball_l").gameObject;
        ball_c = this.transform.Find("ball_c").gameObject;
        ball_r = this.transform.Find("ball_r").gameObject;

        ball_l.SetActive(parameterScript.condition == Parameters.Condition.Left);
        ball_c.SetActive(parameterScript.condition == Parameters.Condition.Center);
        ball_r.SetActive(parameterScript.condition == Parameters.Condition.Right);
    }

    public void TouchCallBack(float touchTime)
    {
        lock (lastTouchLock)
        {
            lastTouch = touchTime;
        }
    }

    public float GetLastTouch()
    {
        lock (lastTouchLock)
        {
            return lastTouch;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
