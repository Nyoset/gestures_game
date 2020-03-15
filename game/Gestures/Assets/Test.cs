using System.Collections;
using System.Collections.Generic;
using IronPython.Hosting;
using UnityEngine;

public class Test : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        var engine = Python.CreateEngine();
        ICollection<string> searchPaths = engine.GetSearchPaths();

        //Path to the folder of greeter.py
        searchPaths.Add("/Users/marcbasquens/Desktop/gestures/game/Gestures/Assets");
        //Path to the Python standard library
        //searchPaths.Add(@"C:\Users\Codemaker\Documents\PythonDemo\Assets\Plugins\Lib\");
        engine.SetSearchPaths(searchPaths);

        dynamic py = engine.ExecuteFile("/Users/marcbasquens/Desktop/gestures/game/Gestures/Assets/test.py");
        dynamic obj = py.Test("Codemaker");
        Debug.Log(obj.display());
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
