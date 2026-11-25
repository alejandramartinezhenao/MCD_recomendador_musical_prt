@echo off
REM Abre la demo del recomendador en el navegador (http://localhost:8501)
cd /d "%~dp0"
py -3.13 -m streamlit run app/app.py
pause
