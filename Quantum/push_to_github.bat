@echo off
setlocal

echo ========================================================
echo    Pushing Quantum AI Project to quantum_capstone
echo ========================================================
echo.

set REPO_URL=https://github.com/gokul2557/quantum_capstone.git

echo Target Repository: %REPO_URL%
echo.

echo [1/5] Initializing Git...
git init
git branch -M main

echo.
echo [2/5] Setting Remote Repository...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo [3/5] Staging all folders and files...
git add .

echo.
echo [4/5] Committing changes...
git commit -m "Initial complete project upload with models and quantum pipeline"

echo.
echo [5/5] Pushing to GitHub...
git push -u origin main --force

echo.
echo ========================================================
echo All files and folders have been uploaded to GitHub!
echo ========================================================
pause
