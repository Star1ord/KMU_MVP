# Быстрый старт

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Настройка структуры данных

Создайте следующую структуру папок:

```
data/
├── audio_wav/
│   ├── 0/    # контроль (положите сюда видео/аудио файлы)
│   └── 1/    # группа риска (положите сюда видео/аудио файлы)
```

## Запуск пайплайна

### Полный пайплайн (рекомендуется)

```bash
python run_full_pipeline.py --limit 3
```

Обработает первые 3 новых файла из `data/audio_wav/`.

### Обработка конкретных файлов

```bash
python run_full_pipeline.py --file-ids file1 file2 file3
```

### Обработка всех файлов

```bash
python run_full_pipeline.py --include-completed
```

## Результаты

После выполнения пайплайна результаты будут в:

- `data/transcripts/` — транскрипты в формате JSON
- `data/segments/` — аудио сегменты и метаданные
- `data/features/merged_features.csv` — финальный датасет с признаками

## Параметры WhisperX

Для Mac M1/M2 (ограниченная память):

```bash
python run_full_pipeline.py --whisper-model small --whisper-compute-type int8 --limit 2
```

Для мощных серверов с GPU:

```bash
python run_full_pipeline.py --whisper-model medium --whisper-device cuda --limit 10
```

## Пропуск этапов

Если какой-то этап уже выполнен, можно его пропустить:

```bash
python run_full_pipeline.py --skip-audio --skip-transcription --limit 5
```

## Помощь

Для просмотра всех доступных параметров:

```bash
python run_full_pipeline.py --help
python extract_audio.py --help
python transcribe_whisperx.py --help
# и т.д.
```

