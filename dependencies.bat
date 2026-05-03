@echo off

echo ================================================
echo   Sri Lanka Wildlife Guide Chatbot
echo   Dependency Installer
echo ================================================
echo.

REM ── Step 1: Check Python 3.11 is available ───────────────────────────────────
echo Checking Python 3.11 installation...
py -3.11 --version
if errorlevel 1 (
    echo ERROR: Python 3.11 not found.
    echo Please install Python 3.11 from https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)
echo Python 3.11 found.
echo.

REM ── Step 2: Create virtual environment using Python 3.11 ─────────────────────
echo Creating virtual environment with Python 3.11...
py -3.11 -m venv venv
echo Virtual environment created.
echo.

REM ── Step 3: Activate virtual environment ────────────────────────────────────
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated.
echo.

REM ── Step 4: Upgrade pip ─────────────────────────────────────────────────────
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM ── Step 5: Install dependencies ────────────────────────────────────────────
echo Installing dependencies...
echo.

pip install tensorflow
pip install nltk
pip install numpy

echo.
echo All dependencies installed successfully.
echo.

REM ── Step 6: Download required NLTK data ─────────────────────────────────────
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
echo NLTK data downloaded.
echo.

REM ── Step 7: Create required folders if they don't exist ─────────────────────
echo Creating project folders...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "src" mkdir src
echo Folders ready.
echo.

echo ================================================
echo   Setup complete!
echo.
echo   Next steps:
echo   1. Place intents.json inside the data folder
echo   2. Run: python src/modelV1.py    to train
echo   3. Run: python src/main.py       to chat
echo ================================================
echo.
pause