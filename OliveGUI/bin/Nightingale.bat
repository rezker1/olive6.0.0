: Note that this is very version dependant, and should be made more general.
: Also assumes bat and jar are in the same directory, as it is on Windows.
java -cp %~dp0\..\repo\easls-5.7.1.1.jar com.sri.speech.forensicgui.ForensicGUISwing %* --workflowDir ..\..\oliveAppData\workflows\ -c ..\config\nightingale_config.xml
