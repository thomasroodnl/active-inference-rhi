using UnityEngine;
using MLAgents;

/// <summary>
/// Class representing the 'rubber' body, supporting the different experimental conditions
/// </summary>
public class RubberArmController : MonoBehaviour
{
    /// <summary>
    /// Parameters allowing the joints to be added from the environment.
    /// </summary>

    [Tooltip("Left shoulder joint")]
    public GameObject leftShoulderJoint;

    [Tooltip("Left upper arm joint")]
    public GameObject leftUpperArmJoint;

    [Tooltip("Left elbow joint")]
    public GameObject leftElbowJoint;

    [Tooltip("Middle of the hand")]
    public GameObject middleHand;

    [Tooltip("Parameters object")]
    public GameObject parameterObject;

    /// <summary>
    /// Joint controller objects for the joints.
    /// </summary>
    private JointController leftShoulder;
    private JointController leftUpperArm;
    private JointController leftElbow;
    private Parameters parameterScript;
    private System.Random rand = new System.Random();


    void Start()
    {
        Academy.Instance.OnEnvironmentReset += () =>
        {
            Reset();
        };
        // Initialize joint controllers
        leftShoulder = new JointController("Left shoulder", leftShoulderJoint, new Vector3(-999, -999, -50), new Vector3(999, 999, 50));
        leftUpperArm = new JointController("Left upper arm", leftUpperArmJoint, new Vector3(-999, -999, -999), new Vector3(999, 999, 999));
        leftElbow = new JointController("Left elbow", leftElbowJoint, new Vector3(-50, -999, -999), new Vector3(50, 999, 999));
        parameterScript = parameterObject.GetComponent<Parameters>();

        //leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, -10f));
        //leftElbow.SetRelativeJointAngles(new Vector3(-5f, 0, 0));
        // Configure joint angles based on condition
        setArmCondition();
    }

    /// <summary>
    /// Sets joint angles such that the rubber arm is in the 'left' position (hand is 15cm left of center, 45cm left of body midline)
    /// </summary>
    public void setLeftCondition()
    {
        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, -10.924f));
        leftElbow.SetRelativeJointAngles(new Vector3(-16.93f, 0, 0));
    }

    /// <summary>
    /// Sets joint angles such that the rubber arm is in the 'right' position (hand is 15cm right of center, 15cm left of body midline)
    /// </summary>
    public void setRightCondition()
    {
        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, 16.296f));
        leftElbow.SetRelativeJointAngles(new Vector3(11.99f, 0, 0));
       // leftUpperArm.SetRelativeJointAngles(new Vector3(11.99f, 0, 0));
    }

    public void setRandomReachCloseCondition()
    {
        float shoulderAngle = (1 + 7 * ((float) rand.NextDouble())) * (-2 * rand.Next(2) + 1);
        float elbowAngle = (1 + 7 * ((float)rand.NextDouble())) * (-2 * rand.Next(2) + 1);
        
        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, shoulderAngle));
        leftElbow.SetRelativeJointAngles(new Vector3(elbowAngle, 0, 0));
    }

    public void setRandomReachFarCondition()
    {
        float shoulderAngle = (8 + 10 * ((float)rand.NextDouble())) * (-2 * rand.Next(2) + 1);
        float elbowAngle = (8 + 10 * ((float)rand.NextDouble())) * (-2 * rand.Next(2) + 1);

        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, shoulderAngle));
        leftElbow.SetRelativeJointAngles(new Vector3(elbowAngle, 0, 0));
    }

    public float getLeftShoulderZ()
    {
        return leftShoulder.GetRelativeZ();
    }

    public float getLeftElbowX()
    {
        return leftElbow.GetRelativeX();
    }

    public void setLeftShoulderZ(float rotation)
    {
        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, rotation));
    }

    public void setLeftElbowX(float rotation)
    {
        leftElbow.SetRelativeJointAngles(new Vector3(rotation, 0, 0));
    }

    /// <summary>
    /// Reset all joints to their 0 position.
    /// </summary>
    public void resetAllJoints()
    {
        leftShoulder.ResetJoint();
        leftElbow.ResetJoint();
    }

    /// <summary>
    /// Reset all joints to their null position
    /// </summary>
    public void Reset()
    {
        leftShoulder.ResetJoint();
        leftUpperArm.ResetJoint();
        leftElbow.ResetJoint();
        setArmCondition();
    }

    public void setArmCondition()
    {
        switch (parameterScript.condition)
        {
            case Parameters.Condition.Left:
                setLeftCondition();
                break;
            case Parameters.Condition.Right:
                setRightCondition();
                break;
            case Parameters.Condition.RandReachClose:
                setRandomReachCloseCondition();
                break;
            case Parameters.Condition.RandReachFar:
                setRandomReachFarCondition();
                break;
            case Parameters.Condition.Center:
                resetAllJoints();
                break;
        }
    }

    public GameObject getMiddleHand()
    {
        return middleHand;
    }

    /// <summary>
    /// Build in void that runs every fixed update (by default every 0.02 seconds)
    /// </summary>
    private void FixedUpdate()
    {
    }

}