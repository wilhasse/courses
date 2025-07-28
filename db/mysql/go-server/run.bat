@echo off
REM Quick run script for Windows

echo Starting MySQL server...
echo Connect with: mysql -h 127.0.0.1 -P 3306 -u root
echo Press Ctrl+C to stop
echo.

if exist "bin\mysql-server.exe" (
    bin\mysql-server.exe
) else (
    echo Binary not found, building first...
    go build -o bin\mysql-server.exe main.go
    if %ERRORLEVEL% equ 0 (
        bin\mysql-server.exe
    ) else (
        echo Build failed! Run setup.bat first.
        pause
    )
)