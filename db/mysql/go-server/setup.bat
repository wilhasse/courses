@echo off
REM Windows setup script for MySQL server with LMDB

echo ========================================
echo MySQL Server with LMDB - Windows Setup
echo ========================================

REM Check for Go installation
go version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Go is not installed or not in PATH
    echo Please install Go from https://golang.org/
    pause
    exit /b 1
)

echo Go found: 
go version

REM Check for Git (needed for dependencies)
git --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

REM Check for C compiler (needed for CGO)
gcc --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo C compiler found: GCC
    goto compiler_found
)

clang --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo C compiler found: Clang
    goto compiler_found
)

echo Warning: No C compiler found
echo Please install one of the following:
echo   - TDM-GCC: https://jmeubank.github.io/tdm-gcc/
echo   - MinGW-w64: https://www.mingw-w64.org/
echo   - Visual Studio Build Tools
echo.
echo Attempting to continue anyway...

:compiler_found

REM Create directories
if not exist bin mkdir bin
if not exist data mkdir data

REM Download Go dependencies
echo Downloading Go dependencies...
go mod tidy
go mod download

REM Check if LMDB library exists
if exist "lmdb-lib\include\lmdb.h" (
    echo LMDB library found
) else (
    echo LMDB library not found
    echo.
    echo For Windows, you have these options:
    echo   1. Use pre-compiled LMDB libraries (recommended)
    echo   2. Compile LMDB from source using MinGW/MSYS2
    echo   3. Use Docker for development (see docker-compose.yml)
    echo.
    echo Please ensure LMDB libraries are in lmdb-lib\ directory:
    echo   lmdb-lib\include\lmdb.h
    echo   lmdb-lib\lib\lmdb.lib (or liblmdb.a)
    echo   lmdb-lib\lib\lmdb.dll (or liblmdb.dll)
    echo.
    pause
)

REM Try to build
echo Attempting to build...
go build -o bin\mysql-server.exe main.go

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo Build completed successfully!
    echo ========================================
    echo.
    echo To run the server:
    echo   bin\mysql-server.exe
    echo.
    echo To connect:
    echo   mysql -h 127.0.0.1 -P 3306 -u root
    echo.
    echo Available batch files:
    echo   run.bat        - Run the server
    echo   build.bat      - Build the project
    echo   clean.bat      - Clean build artifacts
    echo.
) else (
    echo.
    echo ========================================
    echo Build failed!
    echo ========================================
    echo.
    echo Common solutions:
    echo   1. Ensure LMDB libraries are properly installed
    echo   2. Install a C compiler (GCC, Clang, or MSVC)
    echo   3. Use Docker for a complete build environment
    echo.
    echo For Docker:
    echo   docker-compose up mysql-dev
    echo.
)

pause