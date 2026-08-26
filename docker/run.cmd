@echo off
rem Windows launcher, equivalent to docker/run.sh (which needs bash).
rem Mounts .\data only when it exists, and asks for the GPU only when the
rem daemon can provide one.
rem
rem   docker\run.cmd
rem   set CORALNET_DATA=D:\imagery && docker\run.cmd
rem   set LOCKOUT_LEVEL=1 && docker\run.cmd

setlocal enabledelayedexpansion

if "%IMAGE%"=="" set "IMAGE=coralnet-toolbox:local"
if "%CORALNET_DATA%"=="" set "CORALNET_DATA=%cd%\data"
if "%PORT%"=="" set "PORT=6901"
if "%VNC_USER%"=="" set "VNC_USER=user"
if "%VNC_PW%"=="" set "VNC_PW=password"
if "%LOCKOUT_LEVEL%"=="" set "LOCKOUT_LEVEL=2"

rem Preflight. Docker's own errors for these two cases name an endpoint hash
rem rather than the container in the way, which is not much help.
for /f %%i in ('docker ps -a --filter "name=^^coralnet$" --format "{{.Names}}" 2^>nul') do set "EXISTING=%%i"
if not "%EXISTING%"=="" (
    echo error: a container named 'coralnet' already exists.
    echo        docker rm -f coralnet
    exit /b 1
)

for /f %%i in ('docker ps --filter "publish=%PORT%" --format "{{.Names}}" 2^>nul') do set "HOLDER=%%i"
if not "%HOLDER%"=="" (
    echo error: port %PORT% is already published by container '%HOLDER%'.
    echo        docker rm -f %HOLDER%
    echo        or: set PORT=6902 ^&^& docker\run.cmd
    exit /b 1
)

set "ARGS=--rm -it --name coralnet --shm-size=2g -p %PORT%:6901"
set "ARGS=%ARGS% -e VNC_USER=%VNC_USER% -e VNC_PW=%VNC_PW% -e LOCKOUT_LEVEL=%LOCKOUT_LEVEL%"

rem A bare -v would silently CREATE an empty directory on the host if the path
rem were missing. Only mount what is really there.
if exist "%CORALNET_DATA%\" (
    set "ARGS=!ARGS! -v "%CORALNET_DATA%:/home/kasm-user/data""
    echo data:  %CORALNET_DATA% -^> /home/kasm-user/data
) else (
    echo data:  no directory at %CORALNET_DATA% ^(skipping mount^)
)

docker info --format "{{json .Runtimes}}" 2>nul | findstr /C:"nvidia" >nul
if !errorlevel!==0 (
    set "ARGS=!ARGS! --gpus all"
    echo gpu:   enabled
) else (
    echo gpu:   no nvidia runtime detected, running on CPU
)

echo lock:  LOCKOUT_LEVEL=%LOCKOUT_LEVEL% ^(1=desktop, 2=kiosk, 3=unimplemented^)
echo open:  https://localhost:%PORT%  ^(user: %VNC_USER%^)
echo.

docker run !ARGS! %IMAGE%
endlocal
