@echo off
cd /d D:\Projects\QuantForge_Test
call .venv\Scripts\activate
python -m quantforge.run_daily
pause
