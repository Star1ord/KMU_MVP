# KMU MVP

Мультимодальный MVP для оценки риска по пользовательскому медиа, в первую очередь по видео.

## Current Start

### 0) Runtime requirements

- Python `3.10`
- зависимости для локального запуска: `requirements.runtime.txt`

Установка:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.runtime.txt
```

### 1) Full start (recommended)

```bash
python run_all.py --auto-yes
```

### 2) Start separately

Frontend:

```bash
cd web_ui
python -m http.server 5500
```

Backend:

```bash
py -3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger:
`http://127.0.0.1:8000/docs`

OpenAPI request documentation:
`docs/openapi.md`

UI:
`http://127.0.0.1:5500`

### 3) Docker start

```bash
docker compose up --build
```

Если у вас старый Docker CLI:

```bash
docker-compose up --build
```

Сервисы:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5500`

Примечания:

- `models/` монтируется в контейнер backend в режиме read-only.
- `data/` остаётся на хосте и используется для промежуточных артефактов.
- первая сборка может быть долгой из-за ML-зависимостей

### Current working UI modes

- All models (NLP + CV + CV+Audio)
- NLP only
- CV only
- CV+Audio only
- 4. Deception detection
- 5. Emotion audio+video
- 6. Anomaly: audio / video / text

## Current Progress (February 25, 2026)

This is the latest project state as of February 25, 2026.

### Done

- Frontend UI was rebuilt into a dashboard with:
  - dynamic endpoint modes (radio options from config),
  - summary + model metrics table,
  - request metadata block,
  - raw JSON viewer,
  - multiple charts (scores, statuses, timeline, distribution).
- `CV` mode is now available on frontend and mapped to `/test/cv`.
- `/predict` now supports combined execution with `NLP + CV + CV+Audio`.
- `/test/nlp` is isolated from CV/CV+Audio path to keep NLP testing faster.
- `CV+Audio` probability display on frontend was fixed for tiny values
  (scientific notation instead of showing `0` after rounding).
- New test endpoints are integrated into UI:
  - `/test/deception`
  - `/test/emotion-av`
  - `/test/anomaly/audio`
  - `/test/anomaly/video`
  - `/test/anomaly/text`

### Backend stabilization done today

- Added fallback logic for CV/CV+Audio runtime issues (including protobuf-related import problems).
- Added lightweight CV extraction fallback path to avoid hard failures when heavy stack is unavailable.
- Added configurable runtime options in prediction routes for easier testing.

### Current known limitations

- Deception / Emotion / Anomaly endpoints are currently proxy/rule-based test implementations
  (not separate production-trained models yet).
- Full combined mode can still be slow on long videos.
- Some old text below in this README is historical and may not reflect the latest state.

## Local Helper Commands

Проверка на своём видео:

```bash
python examples/run_sample_inference.py path\to\video.mp4
python test_endpoints.py path\to\video.mp4
```

Если положить тестовый файл в `examples/sample_data/`, эти скрипты смогут подобрать его автоматически.

## Описание проекта

Система — это мультимодальная платформа оценки суицидального риска по пользовательскому медиа, в первую очередь по видео.

Пользователь загружает видео. Система выполняет полный препроцессинг: извлекает аудио, распознаёт речь и получает текст (ASR → transcript), а также извлекает кадры и изображения лица (face images). Далее эти данные подаются в несколько независимых моделей.

На текущий момент end-to-end подтверждённо работает только NLP-модель: видео → аудио → транскрипт → NLP-анализ → риск-скор. Это проверено на практике и даёт результат.

CV-модель в проекте присутствует (код и/или веса добавлены), но в текущей интеграции она не работает: при запуске пайплайна CV-результат не получается (модель не запускается, не грузится или silently пропускается). Поэтому сейчас общий результат формируется исключительно на базе NLP.

Также существует модель CV + Audio (лицо + звук), но она пока не подключена к основному пайплайну.

В проекте предусмотрено до 8 основных моделей и финальная 9-я модель — агрегатор (fusion / overall model), которая объединяет сигналы от всех остальных моделей и выдаёт финальный результат.

Финальный результат системы:

- числовой риск-скор
- финальный вердикт

где:

Control (CONTROL / negative) = 0
Risk (RISK / positive / experimental group) = 1

Часть моделей работает по тексту, часть по аудио, часть по лицу, часть по комбинации модальностей. Некоторые аудио-модели принимают только сегменты длительностью от 3 до 13 секунд.

Четвёртая модель — Deception Detection.
Она принимает аудио сегменты длиной 3–13 секунд и возвращает временные ряды по аудио плюс вердикт (positive / negative).

Пятая модель — Emotion Speech & Face.
Она принимает аудио (3–13 секунд) и изображения лица. На выходе возвращает временные аудиоряды (эмоциональные траектории) и итоговый вердикт.

Далее идут аномальные модели.

Anomaly NLP принимает текст (например ответы на открытые вопросы, возможно с изображениями) и возвращает только вердикт.

Anomaly Audio принимает аудио 3–13 секунд и возвращает вердикт.

Anomaly CV принимает изображения лица (face images) и возвращает вердикт.

Все эти модели работают независимо друг от друга и формируют свои собственные результаты: временные ряды, скоры и/или бинарные вердикты.

После этого все выходы поступают в 9-ю модель — агрегатор.

Агрегатор принимает результаты всех предыдущих моделей (NLP, CV, CV+Audio, Deception Detection, Emotion Speech & Face, Anomaly NLP, Anomaly Audio, Anomaly CV), объединяет их и вычисляет финальный риск-скор и финальный класс.

Агрегатор обязан уметь работать даже при частично отсутствующих данных: если не найдено лицо, если нет подходящего аудио сегмента, если какая-то модель временно недоступна. В таких случаях соответствующие сигналы просто пропускаются, а итог считается по тем моделям, которые отработали.

То есть система не должна падать — она всегда должна возвращать результат, используя доступные модальности.

Итоговая логика проекта — это не одна модель, а ансамблевая мультимодальная система:

текст + голос + лицо + эмоции + аномалии
объединяются в общей модели
и дают финальное решение о суицидальном риске (score + verdict)

## Repository Notes

- generated data and caches intentionally do not live in git
- основные runtime-зависимости вынесены в `requirements.runtime.txt`
- Docker-конфиг добавлен как дополнительный способ запуска, а не как замена обычному локальному старту
