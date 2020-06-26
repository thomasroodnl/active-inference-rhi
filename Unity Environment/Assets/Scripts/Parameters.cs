using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

// ===============================
// AUTHOR: Thomas Rood
// PURPOSE: Class functioning as central parameter configuration point.
// ===============================
public class Parameters : MonoBehaviour
{
    /// <summary>
    /// The experimental condition to use.
    /// </summary>
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

    /// <summary>
    /// The arm that is visible to the agent's camera.
    /// </summary>
    public enum VisibleArm
    {
        realArm,
        rubberArm,
        Python_configured
    }

    /// <summary>
    /// The type of stimulation applied.
    /// </summary>
    public enum Stimulation
    {
        synchronous,
        asynchronous,
        Python_configured
    }

    [Tooltip("Experimental condition")]
    public Condition conditionSetMe;

    [Tooltip("Visible arm")]
    public VisibleArm visibleArmSetMe;

    [Tooltip("Stimulation type")]
    public Stimulation stimulationSetMe;

    public Condition condition;

    public VisibleArm visibleArm;

    public Stimulation stimulation;

    private GameObject[] cameraObjects;

    /// <summary>
    /// Start is called before the first frame update.
    /// Configures ApplyParameters() to be executed at reset.
    /// Executes ApplyParameters().
    /// </summary>

    void Start()
    {
        Academy.Instance.OnEnvironmentReset += () =>
        {
            ApplyParameters();
        };

        ApplyParameters();
    }

    /// <summary>
    /// Applies the provided parameters.
    /// </summary>
    void ApplyParameters()
    {
        setCondition();
        setVisibleArm();
        setStimulation();
    }

    /// <summary>
    /// Applies the condition parameter. 
    /// If conditionSetMe == Condition.Python-configured, the condition supplied by the Python script is used.
    /// </summary>
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

    /// <summary>
    /// Applies the visible arm parameter.
    /// Updates the camera's culling mask to match the parameter.
    /// </summary>
    private void setVisibleArm()
    {
        if (visibleArmSetMe == VisibleArm.Python_configured)
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

    /// <summary>
    /// Applies the stimulation parameter.
    /// If stimulationSetMe == Stimulation.Python-configured, the condition supplied by the Python script is used.
    /// </summary>
    private void setStimulation()
    {
        if (stimulationSetMe == Stimulation.Python_configured)
        {
            stimulation = (Stimulation)(int)Academy.Instance.FloatProperties.GetPropertyWithDefault("stimulation", 0f);
        }
        else
        {
            stimulation = stimulationSetMe;
        }
    }
}

