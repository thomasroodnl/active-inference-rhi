using System;
using UnityEngine;
using TMPro;

// ===============================
// AUTHOR: Thomas Rood
// PURPOSE: Class that measures the distance between two objects and prints them to a TextMeshPro object.
// ===============================
public class DistancePrint : MonoBehaviour
{

    [Tooltip("Object 1")]
    public GameObject objectOne;

    [Tooltip("Object 2")]
    public GameObject objectTwo;

    private TextMeshPro tm;

    // Start is called before the first frame update
    void Start()
    {
        tm = GetComponent<TextMeshPro>();
    }

    // Update is called once per frame
    void Update()
    {
        // Set the text to the current distances
        tm.text = "Absolute distance: " + Math.Round(100 * Vector3.Distance(objectOne.transform.position, objectTwo.transform.position),2) + "cm" + 
                  "\nX distance:            " + Math.Round(100 * Math.Abs(objectOne.transform.position.x - objectTwo.transform.position.x),2) + "cm" +
                  "\nY distance:            " + Math.Round(100 * Math.Abs(objectOne.transform.position.y - objectTwo.transform.position.y),2) + "cm" +
                  "\nZ distance:            " + Math.Round(100 * Math.Abs(objectOne.transform.position.z - objectTwo.transform.position.z),2) + "cm";
    }
}
