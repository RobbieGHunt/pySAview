' Silent VBScript Launcher for SuperADAM NeXus Explorer
' Runs pythonw.exe completely silently without any flashing CMD terminal window.

Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
rootDir = fso.GetParentFolderName(scriptDir)

Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = rootDir

pythonwPath = rootDir & "\.venv\Scripts\pythonw.exe"

If fso.FileExists(pythonwPath) Then
    cmd = """" & pythonwPath & """ -m sa_nxs_explorer"
Else
    cmd = "pythonw.exe -m sa_nxs_explorer"
End If

' 0 = Hide window completely, False = Do not block execution
shell.Run cmd, 0, False
