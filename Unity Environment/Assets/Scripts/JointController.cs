using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// ===============================
// AUTHOR: Thomas Rood
// PURPOSE: Class encapsulating a body joint allowing for easy manipulation of the joint.
// ===============================
public class JointController
{
    private readonly string name;
    private readonly GameObject joint;
    private Vector3 startingJointAngles;
    private Vector3 relativeJointAngles;
    private Vector3 minAngles;
    private Vector3 maxAngles;

    /// <summary>
    /// Initialize the JointController.
    /// </summary>
    /// <param name="name">Joint name for debugging</param>
    /// <param name="joint">The GameObject representing the joint</param>
    /// <param name="minAngles">Minimum value of the joint angles allowed</param>
    /// <param name="maxAngles">Maximum value of the joint angles allowed</param>
    public JointController(string name, GameObject joint, Vector3 minAngles, Vector3 maxAngles)
    {
        this.name = name;
        this.joint = joint;
        this.startingJointAngles = joint.transform.localEulerAngles;
        this.relativeJointAngles = new Vector3(0f, 0f, 0f);
        this.minAngles = minAngles;
        this.maxAngles = maxAngles;
    }

    /// <summary>
    /// Move joint by a certain distance.
    /// </summary>
    /// <param name="deltaX">X distance</param>
    /// <param name="deltaY">Y distance</param>
    /// <param name="deltaZ">Z distance</param>
    public void MoveJoint(float deltaX, float deltaY, float deltaZ)
    {
        SetRelativeJointAngles(relativeJointAngles + new Vector3(deltaX, deltaY, deltaZ));
    }

    /// <summary>
    /// Set joint angles to a certain rotation value.
    /// </summary>
    /// <param name="jointAngles">angles to set</param>
    public void SetRelativeJointAngles(Vector3 jointAngles)
    { 
        if (WithinMinMaxAngles(jointAngles))
        {
            joint.transform.localEulerAngles = jointAngles + startingJointAngles;
            relativeJointAngles = jointAngles;
        }
    }

    /// <summary>
    /// Check whether a set of joint angles does not exceed the min and max angle parameters
    /// </summary>
    /// <param name="angles">Set of joint angles to test</param>
    /// <returns>True if angles is within [minAngles, MaxAngles], false otherwise</returns>
    public bool WithinMinMaxAngles(Vector3 angles)
    {
        return minAngles.x <= angles.x && angles.x <= maxAngles.x &&
               minAngles.y <= angles.y && angles.y <= maxAngles.y &&
               minAngles.z <= angles.z && angles.z <= maxAngles.z;
    }

    /// <summary>
    /// Reset joint angles to original position
    /// </summary>
    public void ResetJoint()
    {
        SetRelativeJointAngles(new Vector3(0f, 0f, 0f));
    }

    /// <summary>
    /// Get the joint's x rotation relative to the initial position.
    /// </summary>
    /// <returns>float: The relative x rotation</returns>
    public float GetRelativeX()
    {
        return GetRelativeJointAngles().x;
    }

    /// <summary>
    /// Get the joint's y rotation relative to the initial position.
    /// </summary>
    /// <returns>float: The relative y rotation</returns>
    public float GetRelativeY()
    {
        return GetRelativeJointAngles().y;
    }

    /// <summary>
    /// Get the joint's z rotation relative to the initial position.
    /// </summary>
    /// <returns>float: The relative z rotation</returns>
    public float GetRelativeZ()
    {
        return GetRelativeJointAngles().z;
    }

    /// <summary>
    /// Get the joint's rotation relative to the initial position.
    /// </summary>
    /// <returns>float: The relative rotation</returns>
    public Vector3 GetRelativeJointAngles()
    {
        return this.relativeJointAngles;
    }
}
