@echo off
echo Starting llama.cpp server...
start "Vivianna Server" "F:\AI\llama.cpp\llama-server.exe" ^
  -m "F:\AI\Models\Qwen3.5-9B-UD-Q4_K_XL.gguf" ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -c 8192 ^
  -ngl 99 ^
  --flash-attn on ^
  --jinja ^
  --chat-template-file "F:\AI\Vivianna\chat_template.jinja" ^
  --reasoning off ^
  -ctk q5_1 ^
  -ctv q5_1 ^
  -lv 5

echo Waiting for server to be ready...
:wait
ping -n 6 127.0.0.1 >nul
curl -s http://127.0.0.1:8080/health >nul 2>&1
if errorlevel 1 goto wait

echo Server ready. Launching Vivianna...
cd /d "F:\AI\Vivianna"
call "F:\AI\Vivianna\venv\Scripts\activate.bat"
chcp 65001 >nul
python main.py
pause