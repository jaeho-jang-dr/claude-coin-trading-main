' run_silent.vbs — 콘솔 창 없이 Python 스크립트 실행
' 사용법: wscript.exe run_silent.vbs "script.py" "args"
'
' 동작 방식:
'   1. pythonw.exe (콘솔 없는 Python) 사용
'   2. PYTHONIOENCODING=utf-8 설정
'   3. stdout/stderr → logs/ 폴더에 리다이렉트
'   4. WScript.Shell.Run(..., 0, True) → 창 숨김 + 완료 대기

Dim fso, shell, projectDir, pythonw, scriptPath, scriptName
Dim args, logDir, logFile, cmd

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
pythonw = projectDir & "\.venv\Scripts\pythonw.exe"

' pythonw.exe 없으면 python.exe 폴백 (창은 뜨지만 실행은 됨)
If Not fso.FileExists(pythonw) Then
    pythonw = projectDir & "\.venv\Scripts\python.exe"
End If

' 인자 파싱
If WScript.Arguments.Count < 1 Then
    WScript.Quit 1
End If

scriptPath = WScript.Arguments(0)

' 절대 경로가 아니면 scripts/ 폴더 기준
If InStr(scriptPath, ":") = 0 And Left(scriptPath, 1) <> "\" Then
    scriptPath = projectDir & "\scripts\" & scriptPath
End If

' 스크립트 파일명 추출 (로그용)
scriptName = fso.GetBaseName(scriptPath)

args = ""
Dim i
For i = 1 To WScript.Arguments.Count - 1
    args = args & " " & WScript.Arguments(i)
Next

' 로그 디렉토리 생성
logDir = projectDir & "\logs"
If Not fso.FolderExists(logDir) Then
    fso.CreateFolder(logDir)
End If

' 로그 파일: logs/{script_name}.log (덮어쓰기 — 최근 실행만 보관)
logFile = logDir & "\" & scriptName & ".log"

' pythonw.exe는 stdout/stderr가 없으므로 cmd /c로 리다이렉트
' cmd /c 자체도 창 숨김(0)이므로 콘솔 안 뜸
shell.CurrentDirectory = projectDir

' 환경변수 설정
shell.Environment("Process")("PYTHONIOENCODING") = "utf-8"
shell.Environment("Process")("PYTHONUTF8") = "1"

cmd = "cmd /c """"" & pythonw & """ -u """ & scriptPath & """" & args & " >> """ & logFile & """ 2>&1"""

' 0 = 창 숨김, True = 완료 대기
shell.Run cmd, 0, True
