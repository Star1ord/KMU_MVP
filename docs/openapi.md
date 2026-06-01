# OpenAPI documentation

Документация для backend API KMU MVP.

## Ссылки

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

Чтобы ссылки открывались, backend должен быть запущен:

```bash
py -3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Если проект запущен через `python run_all.py --auto-yes` или Docker, адрес Swagger остается тем же при стандартном порте `8000`.

## Общий формат запросов

Большинство endpoint'ов принимают видео как `multipart/form-data`:

- поле `file` - обязательный видеофайл, кроме `/predict`, где файл может прийти также raw binary body;
- query-параметры передаются в URL после `?`;
- поддерживаемые контейнеры: `mp4`, `mov`, `mkv`, `webm`, `avi`, `m4v`, `3gp`, `mts`, `ts`, `flv`, `wmv` и близкие MIME-типы.

Пример multipart-запроса:

```bash
curl -X POST "http://127.0.0.1:8000/test/nlp?skip_transcription=false" ^
  -F "file=@examples/sample_data/sample.mp4"
```

Пример raw binary-запроса для `/predict`:

```bash
curl -X POST "http://127.0.0.1:8000/predict?include_cv=true&include_cv_audio=true" ^
  -H "Content-Type: video/mp4" ^
  -H "X-File-Name: sample.mp4" ^
  --data-binary "@examples/sample_data/sample.mp4"
```

## Endpoint'ы

### `GET /health`

Проверяет, какие тестовые модели загрузились в backend-процесс.

Response:

```json
{
  "loaded": ["nlp", "cv"],
  "total": 2
}
```

### `POST /predict`

Основной end-to-end endpoint. Принимает видео, запускает препроцессинг, NLP и опционально CV/CV+Audio.

Request:

| Параметр | Где | Тип | По умолчанию | Описание |
| --- | --- | --- | --- | --- |
| `file` | form-data | file | optional | Видео для multipart upload. Если файл не передан, endpoint ожидает raw binary body. |
| `skip_transcription` | query | boolean | `false` | Использовать готовые текстовые артефакты, если они есть, и не запускать ASR повторно. |
| `label` | query | integer | `null` | Опциональная эталонная метка: `0` control, `1` experimental/risk. |
| `include_cv` | query | boolean | `true` | Запускать CV-модальность. |
| `include_cv_audio` | query | boolean | `true` | Запускать CV+Audio-модальность. |
| `video_sample_every` | query | integer | `8` | Брать каждый N-й кадр для CV. Минимум `1`. |
| `cv_audio_sample_rate` | query | number | `0.25` | Частота сэмплирования кадров для CV+Audio. |

Response: форматированный результат общего pipeline. Обычно содержит `success`, итоговый score/verdict, результаты отдельных моделей и служебные metadata. Точная структура может расширяться по мере развития pipeline.

### `POST /test/nlp`

Быстрый endpoint только для NLP-сценария: видео -> аудио -> транскрипт -> NLP risk score.

Request:

| Параметр | Где | Тип | По умолчанию | Описание |
| --- | --- | --- | --- | --- |
| `file` | form-data | file | required | Видео с речью. |
| `skip_transcription` | query | boolean | `false` | Не запускать ASR повторно, если текст уже доступен. |
| `label` | query | integer | `null` | Опциональная эталонная метка: `0` control, `1` experimental/risk. |

Response содержит:

- `success` - успешно ли отработал pipeline;
- `prediction` - `0` control или `1` experimental/risk;
- `prediction_label` - текстовая метка класса;
- `risk_score` и `risk_level`;
- `transcript`, `full_text`, `raw_text`;
- `segments` - сегменты распознанной речи.

### `POST /test/cv`

Endpoint только для CV-модели. Принимает видео, извлекает визуальные фичи и возвращает результат video model.

Request:

| Параметр | Где | Тип | По умолчанию | Описание |
| --- | --- | --- | --- | --- |
| `file` | form-data | file | required | Видео для анализа кадров и лица. |
| `sample_every` | query | integer | `8` | Брать каждый N-й кадр. Минимум `1`. |

Response обычно содержит `success`, `probability`, `prediction`, `prediction_label`, `risk_level`, а также путь к временному CSV с CV-фичами для отладки.

### `POST /predict/cv-audio`

Endpoint для совместной face/audio модели.

Request:

| Параметр | Где | Тип | По умолчанию | Описание |
| --- | --- | --- | --- | --- |
| `file` | form-data | file | required | Видео с видимым лицом и аудиодорожкой. |
| `threshold` | query | number | `0.5` | Порог классификации для класса `1`. Диапазон `0..1`. |
| `sample_rate` | query | number | `0.25` | Частота сэмплирования кадров. |
| `use_vgg` | query | boolean | `false` | Включить VGG16 visual features, если runtime это поддерживает. |

Response содержит `success`, `probability`, `prediction`, `prediction_label`, `risk_level` и диагностические поля pipeline.

### `POST /test/deception`

MVP proxy endpoint для deception-style анализа. Сейчас использует динамику NLP segment risk scores, а не отдельную production deception model.

Request:

| Параметр | Где | Тип | По умолчанию | Описание |
| --- | --- | --- | --- | --- |
| `file` | form-data | file | required | Видео с речевыми сегментами. |
| `skip_transcription` | query | boolean | `false` | Не запускать ASR повторно, если текст уже доступен. |
| `label` | query | integer | `null` | Опциональная эталонная метка: `0` control, `1` experimental/risk. |

Response содержит `deception_score`, `prediction`, `prediction_label`, `risk_level`, `segments_used`, `time_series` и компоненты расчета.

### `POST /test/emotion-av`

MVP proxy endpoint для audio/video emotion анализа. Использует распределение визуальных эмоций и, если доступно, CV+Audio probability.

Request:

| Параметр | Где | Тип | По умолчанию | Описание |
| --- | --- | --- | --- | --- |
| `file` | form-data | file | required | Видео с лицом и опциональной аудиодорожкой. |
| `sample_every` | query | integer | `8` | Брать каждый N-й кадр для emotion features. |
| `sample_rate` | query | number | `0.25` | Частота сэмплирования для дополнительного CV+Audio scoring. |

Response содержит `dominant_emotion`, `emotion_distribution`, `negative_ratio`, `cv_audio_probability`, `emotion_score`, `prediction`, `prediction_label`, `risk_level`.

### `POST /test/anomaly/text`

MVP proxy endpoint для текстовых аномалий. Видео транскрибируется, затем оцениваются лексические признаки.

Request:

| Параметр | Где | Тип | По умолчанию | Описание |
| --- | --- | --- | --- | --- |
| `file` | form-data | file | required | Видео с речью. |
| `skip_transcription` | query | boolean | `false` | Не запускать ASR повторно, если текст уже доступен. |
| `label` | query | integer | `null` | Опциональная эталонная метка: `0` control, `1` experimental/risk. |

Response содержит `modality: "text"`, `anomaly_score`, `prediction`, `prediction_label`, `risk_level` и `metrics` (`word_count`, `lexical_diversity`, `repetition_ratio`, `punctuation_ratio`).

### `POST /test/anomaly/audio`

MVP proxy endpoint для аудио-аномалий. Использует динамику сегментных scores и длительностей.

Request:

| Параметр | Где | Тип | По умолчанию | Описание |
| --- | --- | --- | --- | --- |
| `file` | form-data | file | required | Видео с аудиосегментами. |
| `skip_transcription` | query | boolean | `false` | Не запускать ASR повторно, если текст уже доступен. |
| `label` | query | integer | `null` | Опциональная эталонная метка: `0` control, `1` experimental/risk. |

Response содержит `modality: "audio"`, `anomaly_score`, `prediction`, `prediction_label`, `risk_level`, `time_series` и `metrics`.

### `POST /test/anomaly/video`

MVP proxy endpoint для видео-аномалий. Извлекает CV-индикаторы и при доступности CV-модели добавляет ее probability.

Request:

| Параметр | Где | Тип | По умолчанию | Описание |
| --- | --- | --- | --- | --- |
| `file` | form-data | file | required | Видео для визуального анализа. |
| `sample_every` | query | integer | `8` | Брать каждый N-й кадр. Минимум `1`. |

Response содержит `modality: "video"`, `anomaly_score`, `prediction`, `prediction_label`, `risk_level`, `cv_probability` и `indicators`.

## Статусы ошибок

- `400 Bad Request` - неподдерживаемый тип файла или пустой raw body.
- `422 Unprocessable Entity` - не хватает обязательного поля `file` или query-параметр не прошел валидацию.
- `500 Internal Server Error` - ошибка обработки внутри pipeline.
- `503 Service Unavailable` - нужный pipeline или модель недоступны на сервере.

## Что отправить как ссылку

Для разработчиков и тестировщиков:

`http://127.0.0.1:8000/docs`

Для просмотра статической схемы:

`http://127.0.0.1:8000/openapi.json`

Для человекочитаемого описания запросов в репозитории:

`docs/openapi.md`
