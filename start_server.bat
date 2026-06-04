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

pause