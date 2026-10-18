@echo off
echo ========================================================
echo Requesting Administrator privileges for OpenSSH setup...
echo ========================================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { Start-Process powershell.exe -Verb RunAs -ArgumentList @('-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', '%~dp0setup_openssh_admin.ps1') }"
