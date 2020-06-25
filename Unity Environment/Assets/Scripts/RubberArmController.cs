using UnityEngine;
using MLAgents;

// ===============================
// AUTHOR: Thomas Rood
// PURPOSE: Class that controls the rubber arm.
// ===============================
public class RubberArmController : MonoBehaviour
{
    /// <summary>
    /// Parameters that allow the specification of the joint angles
    /// </summary>
    [Tooltip("Left shoulder joint")]
    public GameObject leftShoulderJoint;

    [Tooltip("Left upper arm joint")]
    public GameObject leftUpperArmJoint;

    [Tooltip("Left elbow joint")]
    public GameObject leftElbowJoint;

    /// <summary>
    /// Parameters that allow for distance measurement
    /// </summary>
    [Tooltip("Middle of the hand")]
    public GameObject middleHand;

    /// <summary>
    /// Parameter that allows reading the environment parameters
    /// </summary>
    [Tooltip("Parameters object")]
    public GameObject parameterObject;

    /// <summary>
    /// Joint controller objects for the joints
    /// </summary>
    private JointController leftShoulder;
    private JointController leftUpperArm;
    private JointController leftElbow;

    private Parameters parameterScript;
    private System.Random rand = new System.Random();

    // Start is called before the first frame update
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

        // Configure joint angles based on condition
        setArmCondition();
    }

    /// <summary>
    /// Sets joint angles such that the rubber arm is in the 'left' position (hand is 15cm left of center, 45cm left of body midline).
    /// </summary>
    public void setLeftCondition()
    {
        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, -10.924f));
        leftElbow.SetRelativeJointAngles(new Vector3(-16.93f, 0, 0));
    }

    /// <summary>
    /// Sets joint angles such that the rubber arm is in the 'right' position (hand is 15cm right of center, 15cm left of body midline).
    /// </summary>
    public void setRightCondition()
    {
        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, 16.296f));
        leftElbow.SetRelativeJointAngles(new Vector3(11.99f, 0, 0));
    }

    /// <summary>
    /// Sets joint angles such that the rubber arm is at a random rotation in the range +-[1, 8)
    /// </summary>
    public void setRandomReachCloseCondition()
    {
        float shoulderAngle = (1 + 7 * ((float) rand.NextDouble())) * (-2 * rand.Next(2) + 1);
        float elbowAngle = (1 + 7 * ((float)rand.NextDouble())) * (-2 * rand.Next(2) + 1);
        
        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, shoulderAngle));
        leftElbow.SetRelativeJointAngles(new Vector3(elbowAngle, 0, 0));
    }

    /// <summary>
    /// Sets joint angles such that the rubber arm is at a random rotation in the range +-[8, 18)
    /// </summary>
    public void setRandomReachFarCondition()
    {
        float shoulderAngle = (8 + 10 * ((float)rand.NextDouble())) * (-2 * rand.Next(2) + 1);
        float elbowAngle = (8 + 10 * ((float)rand.NextDouble())) * (-2 * rand.Next(2) + 1);

        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, shoulderAngle));
        leftElbow.SetRelativeJointAngles(new Vector3(elbowAngle, 0, 0));
    }

    /// <summary>
    /// Get the shoulder's z rotation relative to the initial position.
    /// </summary>
    /// <returns>float: The relative z rotation</returns>
    public float getRelativeLeftShoulderZ()
    {
        return leftShoulder.GetRelativeZ();
    }

    /// <summary>
    /// Get the elbow's x rotation relative to the initial position.
    /// </summary>
    /// <returns>float: The relative x rotation</returns>
    public float getRelativeLeftElbowX()
    {
        return leftElbow.GetRelativeX();
    }

    /// <summary>
    /// Set the shoulder's z rotation (relative to the initial position).
    /// </summary>
    public void setRelativeLeftShoulderZ(float rotation)
    {
        leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, rotation));
    }

    /// <summary>
    /// Set the elbow's x rotation (relative to the initial position).
    /// </summary>
    public void setRelativeLeftElbowX(float rotation)
    {
        leftElbow.SetRelativeJointAngles(new Vector3(rotation, 0, 0));
    }

    /// <summary>
    /// Reset all joints to their initial position.
    /// </summary>
    public void resetAllJoints()
    {
        leftShoulder.ResetJoint();
        leftUpperArm.ResetJoint();
        leftElbow.ResetJoint();
    }

    /// <summary>
    /// Called upon reset of the environment. Resets the joint positions and reconfigures the arm condition.
    /// </summary>
    public void Reset()
    {
        resetAllJoints();
        setArmCondition();
    }

    /// <summary>
    /// Run the right configuration function to set the arm condition based on the parameter setting.
    /// </summary>
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

    /// <summary>
    /// Get the GameObject roughly representing the middle of the hand.
    /// </summary>
    /// <returns></returns>
    public GameObject getMiddleHand()
    {
        return middleHand;
    }
}