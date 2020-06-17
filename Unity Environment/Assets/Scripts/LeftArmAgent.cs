using UnityEngine;
using MLAgents;
using UnityEngine.SceneManagement;

/// <summary>
/// Class representing the agent or 'real' body
/// </summary>
public class LeftArmAgent : Agent
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

    [Tooltip("Head joint")]
    public GameObject headJoint;

    [Tooltip("Middle hand")]
    public GameObject middleHand;

    [Tooltip("Rubber Arm")]
    public GameObject rubberArm;

    [Tooltip("Ball")]
    public GameObject ballHandler;

    [Tooltip("Vib")]
    public GameObject vibHandler;


    /// <summary>
    /// Joint controller objects for the joints.
    /// </summary>
    private JointController leftShoulder;
    private JointController leftUpperArm;
    private JointController leftElbow;
    private JointController head;

    private RubberArmController rubberArmController;
    private BallHandler ballScript;
    private VibHandler vibScript;

    /// <summary>
    /// Turn speed in degrees/second.
    /// </summary>
    private float turnSpeed = 1f;

    /// <summary>
    /// Initial setup, called when the agent is enabled
    /// </summary>
    /// <remarks>
    /// In future versions of ML-agents (> 0.14), InitializeAgent should be replaced by Initialize.
    /// </remarks>
    public override void InitializeAgent()
    {
        base.InitializeAgent();

        // Initialize joint controllers
        leftShoulder = new JointController("Left shoulder", leftShoulderJoint, new Vector3(-999, -999, -999), new Vector3(999, 999, 999)); //new Vector3(-999, -999, -25), new Vector3(999, 999, 25));
        leftUpperArm = new JointController("Left upper arm", leftUpperArmJoint, new Vector3(-999, -999, -999), new Vector3(999, 999, 999));
        leftElbow    = new JointController("Left elbow", leftElbowJoint, new Vector3(-999, -999, -999), new Vector3(999, 999, 999)); //new Vector3(-25, -999, -999), new Vector3(25, 999, 999));
        head = new JointController("Head", headJoint, new Vector3(-999, -999, -999), new Vector3(999, 999, 999));

        rubberArmController = rubberArm.GetComponent<RubberArmController>();
        ballScript = ballHandler.GetComponent<BallHandler>();
        vibScript = vibHandler.GetComponent<VibHandler>();

        //HeadJoint.transform.localPosition = new Vector3(-0.0273f, 0.013f, 0.0357f); // (dataset and network ne1)
        //head.SetRelativeJointAngles(new Vector3(60f, -0f)); // (dataset and network ne1)

        head.SetRelativeJointAngles(new Vector3(15f, -20f)); // (dataset and network 6 and s1, s2)
        //leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, 31.218f));
        //head.SetRelativeJointAngles(new Vector3(12f, -25f)); // (dataset and network 7)
        //head.SetRelativeJointAngles(new Vector3(15f, -22f)); // (dataset and network s3 and s4)
        //head.SetRelativeJointAngles(new Vector3(23f, -17.7f)); // 

        //HeadJoint.transform.localPosition = new Vector3(0f, 0.00654f, 0.0103f); // (dataset and network ne1)
        //head.SetRelativeJointAngles(new Vector3(30.854f, -21.931f)); // 


        //HeadJoint.transform.localPosition = new Vector3(0f, 0.00693f, 0.01187f);
        //head.SetRelativeJointAngles(new Vector3(24f, -28f));

    }

    /// <summary>
    /// Perform actions based on a vector of numbers
    /// </summary>
    /// <remarks>
    /// In future versions of ML-agents (> 0.14), AgentAction should be replaced by OnActionReceived.
    /// </remarks>
    /// <param name="vectorAction">The list of actions to take</param>
    public override void AgentAction(float[] vectorAction)
    {
        if(vectorAction[0] == 0f) 
        {
            leftShoulder.MoveJoint(0f, 0f, vectorAction[1] * turnSpeed * Time.fixedDeltaTime);
            leftElbow.MoveJoint(vectorAction[2] * turnSpeed * Time.fixedDeltaTime, 0f, 0f);
        } else
        {
            if(vectorAction[0] == 1f)
            {
                leftShoulder.SetRelativeJointAngles(new Vector3(0f, 0f, vectorAction[1]));
                leftElbow.SetRelativeJointAngles(new Vector3(vectorAction[2], 0f, 0f));
            }
            else
            {
                rubberArmController.setLeftShoulderZ(vectorAction[1]);
                rubberArmController.setLeftElbowX(vectorAction[2]);
            }
        }
    }

    /// <summary>
    /// Read inputs from the keyboard and convert them to a list of actions.
    /// This is called only when the player wants to control the agent and has set
    /// Behavior Type to "Heuristic Only" in the Behavior Parameters inspector.
    /// </summary>
    /// <returns>A vectorAction array of floats that will be passed into <see cref="AgentAction(float[])"/></returns>
    public override float[] Heuristic()
    {
        float shoulderRotate = 0f;
        if (Input.GetKey(KeyCode.A))
        {
            // turn left
            shoulderRotate = -0.5f;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            // turn right
            shoulderRotate = 0.5f;
        }

        float elbowRotate = 0f;
        if (Input.GetKey(KeyCode.W))
        {
            // turn left
            elbowRotate = -0.5f;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            // turn right
            elbowRotate = 0.5f;
        }


        // Put the actions into an array and return
        return new float[] { 0f, shoulderRotate, elbowRotate };
    }

    /// <summary>
    /// Reset the agent and area
    /// </summary>
    /// <remarks>
    /// In future versions of ML-agents (> 0.14), AgentReset should be replaced by OnEpisodeBegin.
    /// </remarks>
    public override void AgentReset()
    {
        leftShoulder.ResetJoint();
        leftUpperArm.ResetJoint();
        leftElbow.ResetJoint();
        //InitializeAgent();
        //leftShoulder.SetRelativeJointAngles(new Vector3(0, 0, 31.218f));
    }


    /// <summary>
    /// Collect all non-Raycast observations
    /// </summary>
    public override void CollectObservations()
    {
        // Shoulder extension/flexion
        AddVectorObs(leftShoulder.GetRelativeZ());

        // Upper arm twist
        // AddVectorObs(leftUpperArm.GetRelativeY());
        
        // Elbow extension/flexion
        AddVectorObs(leftElbow.GetRelativeX());

        // Hand touch state (visual)
        AddVectorObs(ballScript.GetLastTouch());

        // Hand touch state (tactile)
        AddVectorObs(vibScript.GetLastVib());

        // Current Time
        AddVectorObs(Time.time);

        // Absolute hand distance error for logging
        AddVectorObs(Vector3.Distance(middleHand.transform.position, rubberArmController.getMiddleHand().transform.position));

        // Horizontal distance
        AddVectorObs(middleHand.transform.position.z - rubberArmController.getMiddleHand().transform.position.z);

        // Shoulder extension/flexion
        AddVectorObs(rubberArmController.getLeftShoulderZ());

        // Upper arm twist
        // AddVectorObs(leftUpperArm.GetRelativeY());

        // Elbow extension/flexion
        AddVectorObs(rubberArmController.getLeftElbowX());

        //Debug.Log(Vector3.Distance(middleHand.transform.position, rubberArmController.getMiddleHand().transform.position))
        // Head horizontal rotation
        //AddVectorObs(head.GetRelativeY());

        // Head vertical rotaiton
        //AddVectorObs(head.GetRelativeX());

    }

    /// <summary>
    /// Build in void that runs every fixed update (by default every 0.02 seconds)
    /// </summary>
    private void FixedUpdate()
    {
        //Quaternion myRotation = Quaternion.identity;
        //myRotation.eulerAngles = new Vector3(0, -90, 180);
        //leftShoulderJoint.transform.rotation = myRotation;
        //Debug.Log("Shoulder " + leftShoulderJoint.transform.rotation.eulerAngles + "   Elbow " + leftElbowJoint.transform.localEulerAngles);
        //leftElbow.MoveJoint(1f * turnSpeed * Time.fixedDeltaTime, 0, 0
        //leftElbow.SetRelativeJointAngles(new Vector3(0, 0, -10));
        //leftElbow.SetRelativeJointAngles(new Vector3(40, 0, 0));
    }

}