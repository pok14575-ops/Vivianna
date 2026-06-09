@echo off
cd /d "%~dp0"

REM --- Load machine-specific paths from .env (gitignored). See .env.example. ---
if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do set "%%A=%%B"
)
REM Fall back to the default in-repo layout if .env didn't set these.
if not defined LLAMA_EXE set "LLAMA_EXE=%~dp0llama.cpp\llama-server.exe"
if not defined VIVIANNA_LLM_MODEL set "VIVIANNA_LLM_MODEL=%~dp0models\Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf"

echo Starting llama.cpp server (MTP speculative decoding)...
start "Vivianna Server" "%LLAMA_EXE%" ^
  -m "%VIVIANNA_LLM_MODEL%" ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -c 8192 ^
  -ngl 99 ^
  --flash-attn on ^
  --jinja ^
  --chat-template-file "%~dp0chat_template.jinja" ^
  --reasoning off ^
  -ctk q5_1 ^
  -ctv q5_1 ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 2 ^
  --parallel 1 ^
  -lv 5

echo Waiting for server to be ready...
:wait
ping -n 6 127.0.0.1 >nul
curl -s http://127.0.0.1:8080/health >nul 2>&1
if errorlevel 1 goto wait

echo Server ready. Launching Vivianna...
call "%~dp0venv\Scripts\activate.bat"
chcp 65001 >nul
python main.py
pause
