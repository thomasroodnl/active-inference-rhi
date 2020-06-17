using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class Parameters : MonoBehaviour
{
    public enum Condition
    {
        Left,
        Center,
        Right,
        RandReachClose,
        RandReachFar,
        _Break,
        Python_configured
    }

    public enum Stimulation
    {
        synchronous,
        asynchronous
    }

    public enum VisibleArm
    {
        realArm,
        rubberArm,
        Python_configured
    }

    [Tooltip("Experimental condition")]
    public Condition conditionSetMe;

    [Tooltip("Visible arm")]
    public VisibleArm visibleArmSetMe;

    [Tooltip("Stimulation type")]
    public Stimulation stimulation;

    public Condition condition;

    public VisibleArm visibleArm;

    private GameObject[] cameraObjects;

    // Start is called before the first frame update
    void Start()
    {
        Academy.Instance.OnEnvironmentReset += () =>
        {
            Reset();
        };

        setCondition();
        setVisibleArm();
    }

    void Reset()
    {
        setCondition();
        setVisibleArm();
    }

    private void setCondition()
    {
        if (conditionSetMe == Condition.Python_configured)
        {
            condition = (Condition)(int)Academy.Instance.FloatProperties.GetPropertyWithDefault("condition", 1f);
        }
        else
        {
            condition = conditionSetMe;
        }
    }

    private void setVisibleArm()
    {
        if (visibleArm == VisibleArm.Python_configured)
        {
            visibleArm = (VisibleArm)(int)Academy.Instance.FloatProperties.GetPropertyWithDefault("visiblearm", 0f);
        }
        else
        {
            visibleArm = visibleArmSetMe;
        }

        cameraObjects = GameObject.FindGameObjectsWithTag("eye_camera");

        foreach (GameObject cameraObject in cameraObjects)
        {
            Camera camera = cameraObject.GetComponent(typeof(Camera)) as Camera;

            switch (visibleArm)
            {
                case VisibleArm.realArm:
                    // Layer 8 (256), 9 (512) and 12 (4096) bitmask
                    camera.cullingMask = 256 + 512 + 4096;
                    break;
                case VisibleArm.rubberArm:
                    // Layer 8 (256), 10 (1024) and 12 (4096) bitmask
                    camera.cullingMask = 256 + 1024 + 4096;
                    break;
            }
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

}

