---
title: Simpsons Who
emoji: 🍩
colorFrom: yellow
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Simpsons-who? — классификатор персонажей Симпсонов


- Широкий кадр сервис смотрит по частям (слева, по центру, справа) и показывает, где нашёл персонажа,
  плюс других уверенно найденных персонажей.
- Если персонажей несколько, можно обвести нужного мышкой.

Запуск: `pip install -r requirements.txt && uvicorn server:app --port 8000` → http://localhost:8000

API: `POST /api/predict` (multipart: `image`, необязательно `box=x1,y1,x2,y2`), `GET /api/info`, `GET /health`.

Настройки через переменные окружения: `MODEL_DIR` (по умолчанию `./model`), `THREADS`,
`CONFIDENT` (порог «не уверен», 0.5), `ALSO_MIN` (порог «ещё в кадре», 0.6).
