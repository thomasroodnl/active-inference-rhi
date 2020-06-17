using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Class encapsulating a joint and controlling its position and movement
/// </summary>
public class JointController
{
    private readonly string name;
    private readonly GameObject joint;
    private Vector3 startingJointAngles;
    private Vector3 relativeJointAngles;
    private Vector3 minAngles;
    private Vector3 maxAngles;

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
    /// Move joint by a certain distance
    /// </summary>
    /// <param name="deltaX">X distance</param>
    /// <param name="deltaY">Y distance</param>
    /// <param name="deltaZ">Z distance</param>
    public void MoveJoint(float deltaX, float deltaY, float deltaZ)
    {
        SetRelativeJointAngles(relativeJointAngles + new Vector3(deltaX, deltaY, deltaZ));
    }

    public void ApplyTorque(float torqueX, float torqueY, float torqueZ)
    {

    }

    /// <summary>
    /// Set joint angles to a certain value
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
    /// Check whether a set of angles does not exceed the max angle parameter
    /// </summary>
    /// <param name="angles">set of angles to test</param>
    /// <returns>True if angles is within MaxAngles, false otherwise</returns>
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

    public float GetRelativeX()
    {
        return GetRelativeJointAngles().x;
    }

    public float GetRelativeY()
    {
        return GetRelativeJointAngles().y;
    }

    public float GetRelativeZ()
    {
        return GetRelativeJointAngles().z;
    }

    public float GetNormalizedX()
    {
        return ((GetRelativeX() - minAngles.x) / (maxAngles.x - minAngles.x))* 2 - 1;
    }

    public float GetNormalizedY()
    {
        return ((GetRelativeY() - minAngles.y) / (maxAngles.y - minAngles.y)) * 2 - 1;
    }

    public float GetNormalizedZ()
    {
        return ((GetRelativeZ() - minAngles.z) / (maxAngles.z - minAngles.z)) * 2 - 1;
    }


    public Vector3 GetRelativeJointAngles()
    {
        return this.relativeJointAngles;
    }
}
