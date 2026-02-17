# Пайплайн обработки мультимодальных данных речи

Пайплайн для извлечения транскриптов (WhisperX) и акустических признаков (openSMILE eGeMAPS) из аудио/видео файлов.

## Структура пайплайна

Пайплайн состоит из следующих модулей:

### Основные модули

- **`extract_audio.py`** — Подготовка аудио: конвертация видео из папок 0/1 в WAV
- **`transcribe_whisperx.py`** — Транскрипция аудио с WhisperX с поддержкой инкрементальной обработки
- **`segment_audio.py`** — Сегментация аудио на основе транскриптов WhisperX
- **`extract_opensmile_features.py`** — Извлечение акустических признаков с помощью openSMILE
- **`merge_features.py`** — Объединение аудио и текстовых признаков в единый датасет
- **`run_full_pipeline.py`** — Главный скрипт для запуска всего пайплайна на выбранных файлах

### Вспомогательные модули

- **`media_utils.py`** — Утилиты для работы с медиаданными (поиск файлов, определение путей, проверка консистентности)
- **`__init__.py`** — Инициализация пакета

## Требования

### Системные зависимости

- **Python 3.10+**
- **FFmpeg** (для конвертации видео в WAV)
- **openSMILE** (для извлечения акустических признаков)

### Python зависимости

Установите зависимости из `requirements.txt`:

```bash
pip install -r requirements.txt
```

Основные зависимости:
- `whisperx>=3.1.1` — транскрипция аудио
- `pandas>=2.0.0` — обработка данных
- `numpy>=1.24.0` — численные вычисления
- `pydub>=0.25.1` — обработка аудио
- `tqdm>=4.65.0` — прогресс-бары
- `psutil>=5.9.0` — определение системных ресурсов

## Структура данных

Пайплайн ожидает следующую структуру данных:

```
data/
├── audio_wav/
│   ├── 0/    # контроль (обычные пользователи)
│   └── 1/    # потенциально суицидальные пользователи
├── transcripts/        # JSON WhisperX (создаётся автоматически)
│   ├── 0/
│   └── 1/
├── segments/           # сегменты + segments_metadata.csv
│   └── segments_metadata.csv
└── features/
    ├── opensmile_features.csv
    └── merged_features.csv
```

- В папки `data/audio_wav/0` и `data/audio_wav/1` можно класть **и видеоролики (.mp4/.mov/.mkv)**, и готовые WAV.
- При каждом запуске пайплайн пропускает уже обработанные файлы и добавляет новые результаты.

## Использование

### Полный пайплайн

Запуск всего пайплайна на выбранных файлах:

```bash
python run_full_pipeline.py --limit 3
```

Параметры:
- `--file-ids id1 id2` — явно указать список file_id для обработки
- `--limit N` — сколько новых видео обработать за запуск
- `--include-completed` — разрешить переработку уже готовых видео
- `--whisper-model small` — выбрать модель WhisperX (tiny/base/small/medium/large)
- `--whisper-device cuda` — использовать GPU (по умолчанию cpu)
- `--skip-*` — пропустить отдельные стадии (`--skip-audio`, `--skip-transcription`, и т.д.)
- `--aggregate` — дополнительно собрать `features_per_video.csv`

### Отдельные этапы

Каждый этап можно запускать отдельно:

#### 1. Подготовка аудио

```bash
python extract_audio.py --input-dir data/audio_wav --file-ids id1 id2
```

#### 2. Транскрипция

```bash
python transcribe_whisperx.py --input-dir data/audio_wav --output-dir data/transcripts --model small
```

#### 3. Сегментация

```bash
python segment_audio.py --audio-dir data/audio_wav --transcript-dir data/transcripts --output-dir data/segments
```

#### 4. Извлечение признаков openSMILE

```bash
python extract_opensmile_features.py --segments-dir data/segments --output-dir data/features --opensmile-dir ../opensmile
```

#### 5. Объединение признаков

```bash
python merge_features.py --segments-metadata data/segments/segments_metadata.csv --opensmile-features data/features/opensmile_features.csv --output data/features/merged_features.csv
```

## Поток обработки

1. **Подготовка аудио** (`extract_audio.py`): конвертирует видео в WAV (моно, 16 kHz, нормализация)
2. **WhisperX** (`transcribe_whisperx.py`): создаёт JSON транскрипты с таймстемпами в `data/transcripts/<label>/file.json`
3. **Сегментация** (`segment_audio.py`): создаёт WAV-фрагменты и `segments_metadata.csv` (содержит label, относительные пути к аудио/транскрипту)
4. **openSMILE** (`extract_opensmile_features.py`): извлекает eGeMAPS признаки для каждого сегмента
5. **Merge** (`merge_features.py`): объединяет всё в `data/features/merged_features.csv` + добавляет текстовые признаки (speech rate, pronoun_count и т.д.)

## Выходные данные

- `data/segments/segments_metadata.csv` — список сегментов с таймстемпами, текстом, label и путями до файлов
- `data/features/opensmile_features.csv` — eGeMAPS признаки для каждого сегмента
- `data/features/merged_features.csv` — объединённый датасет (включая текстовые признаки)
- `data/features/features_per_video.csv` — (опционально) агрегации по видео (`--aggregate`)

## Особенности

- **Инкрементальная обработка**: повторные запуски пропускают уже обработанные файлы
- **Гибкая фильтрация**: можно обрабатывать конкретные file_id или ограничивать количество файлов
- **Автоматическое определение окружения**: пайплайн автоматически определяет оптимальные параметры для Mac/сервера
- **Обработка ошибок**: каждый этап логирует ошибки и продолжает работу с остальными файлами

## Зависимости между модулями

Все модули используют `media_utils.py` для работы с медиаданными. Модули можно запускать независимо, но рекомендуется использовать `run_full_pipeline.py` для полной обработки.

## Примечания

- Для работы с openSMILE необходимо собрать бинарник `SMILExtract` из исходников openSMILE
- WhisperX автоматически загружает модели при первом запуске (может занять время)
- На Mac M1 рекомендуется использовать `--whisper-model small --whisper-compute-type int8`

