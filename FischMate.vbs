Option Explicit

Dim shell, files, root, python, pythonw, checkCommand, checkResult, command
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
python = root & "\.venv\Scripts\python.exe"
pythonw = root & "\.venv\Scripts\pythonw.exe"

If Not files.FileExists(python) Or Not files.FileExists(pythonw) Then
    MsgBox "FischMate setup is not finished." & vbCrLf & vbCrLf & _
        "Open Command Prompt in this folder and run:" & vbCrLf & _
        "python -m venv .venv" & vbCrLf & _
        ".venv\Scripts\python -m pip install -r requirements.txt", _
        48, "FischMate Setup Required"
    WScript.Quit 1
End If

checkCommand = Chr(34) & python & Chr(34) & " -c " & Chr(34) & _
    "import cv2, dxcam, mss, numpy, yaml" & Chr(34)
checkResult = shell.Run(checkCommand, 0, True)
If checkResult <> 0 Then
    MsgBox "FischMate dependencies are missing or incomplete." & vbCrLf & vbCrLf & _
        "Open Command Prompt in this folder and run:" & vbCrLf & _
        ".venv\Scripts\python -m pip install -r requirements.txt", _
        48, "FischMate Dependencies Required"
    WScript.Quit 1
End If

shell.CurrentDirectory = root
command = Chr(34) & pythonw & Chr(34) & " -m app.main"
shell.Run command, 0, False
