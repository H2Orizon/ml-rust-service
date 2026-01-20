@echo off
echo Starting all services...

start cmd /k "cd python_ml && python train.py"
timeout /t 5

start cmd /k "cd rust_api && cargo run"
timeout /t 5

start cmd /k "cd frontend && streamlit run app.py"

echo All services started.
pause