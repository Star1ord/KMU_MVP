import argparse
import contextlib
import csv
import glob
import io
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[3]
VIDEO_INTEGRATION_DIR = ROOT / "video_integration"
if str(VIDEO_INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEO_INTEGRATION_DIR))

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import realtime_test as _realtime_test
    from realtime_test import HSEMOTION_AVAILABLE, RealtimeAnalyzer


class VideoAnalyzer(RealtimeAnalyzer):
    """
    Offline-анализатор, наследующий все признаки RealtimeAnalyzer,
    но позволяющий отключать тяжелую отрисовку.
    """

    def __init__(self, render_overlay: bool = False, use_emotions: Optional[bool] = None):
        original_hsemotion_available = getattr(_realtime_test, "HSEMOTION_AVAILABLE", False)
        if use_emotions is False:
            _realtime_test.HSEMOTION_AVAILABLE = False

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                super().__init__()
        finally:
            _realtime_test.HSEMOTION_AVAILABLE = original_hsemotion_available

        self.render_overlay = render_overlay
        if use_emotions is not None:
            self.use_emotions = bool(use_emotions and original_hsemotion_available)

    def draw_info(self, frame, avg_ear):
        """Переопределяем, чтобы по умолчанию не рисовать HUD."""
        if self.render_overlay:
            super().draw_info(frame, avg_ear)

    def snapshot_metrics(self, timestamp: float) -> dict:
        """Возвращает словарь с текущими метриками для записи в CSV."""
        return {
            "timestamp": f"{timestamp:.2f}",
            "blink_count": self.blink_count,
            "ear": round(self.ear_history[-1], 4) if self.ear_history else "",
            "head_yaw": round(self.head_yaw, 2),
            "head_pitch": round(self.head_pitch, 2),
            "head_roll": round(self.head_roll, 2),
            "gaze_direction": self.gaze_direction,
            "looking_at_camera": int(self.looking_at_camera),
            "gaze_away_time": round(self.gaze_away_time, 2),
            "head_down": int(self.head_down),
            "head_down_duration": round(self.head_down_duration, 2),
            "head_stability": round(self.head_stability, 2),
            "hands_detected": self.hands_detected,
            "touching_face": int(self.touching_face),
            "face_touch_count": self.face_touch_count,
            "emotion": self.current_emotion,
            "emotion_duration": round(self.emotion_duration, 2),
        }


class EventLogger:
    """Простой JSONL логгер событий."""

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "w", encoding="utf-8")
        self._active: Dict[str, Dict[str, float]] = {}

    def update(self, timestamp: float, event_state: Dict[str, Tuple[bool, Dict]]) -> None:
        for name, (is_active, payload) in event_state.items():
            if is_active:
                if name not in self._active:
                    self._active[name] = {"start": timestamp, "payload": payload}
                else:
                    self._active[name]["payload"] = payload
            elif name in self._active:
                self._emit(name, self._active.pop(name), timestamp)

    def close(self, final_timestamp: Optional[float] = None) -> None:
        if final_timestamp is not None:
            for name in list(self._active.keys()):
                self._emit(name, self._active.pop(name), final_timestamp)
        self._file.close()

    def _emit(self, name: str, info: Dict[str, float], end_time: float) -> None:
        event = {
            "event": name,
            "start": round(info["start"], 2),
            "end": round(end_time, 2),
            "duration": round(end_time - info["start"], 2),
            "details": info.get("payload", {}),
        }
        self._file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._file.flush()


def collect_risk_state(analyzer: VideoAnalyzer) -> Dict[str, Tuple[bool, Dict]]:
    """Возвращает текущее состояние маркеров риска."""
    return {
        "head_down_long": (
            bool(analyzer.head_down and analyzer.head_down_duration >= 3.0),
            {
                "head_pitch": round(analyzer.head_pitch, 2),
                "head_down_duration": round(analyzer.head_down_duration, 2),
            },
        ),
        "gaze_avoidance": (
            bool((not analyzer.looking_at_camera) and analyzer.gaze_away_time >= 5.0),
            {
                "gaze_direction": analyzer.gaze_direction,
                "gaze_away_time": round(analyzer.gaze_away_time, 2),
            },
        ),
        "long_sadness": (
            bool(analyzer.current_emotion == "sad" and analyzer.emotion_duration >= 5.0),
            {
                "emotion_duration": round(analyzer.emotion_duration, 2),
            },
        ),
        "low_head_stability": (
            bool(analyzer.head_stability < 0.5),
            {
                "head_stability": round(analyzer.head_stability, 2),
            },
        ),
    }


def discover_videos(input_dir: str, extensions: Iterable[str]) -> List[str]:
    """Возвращает список видеофайлов в каталоге."""
    videos: List[str] = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    return sorted(videos)


def process_video_file(
    video_path: str,
    output_csv: str,
    sample_every: int = 3,
    render_overlay: bool = False,
    use_emotions: Optional[bool] = None,
    event_log_path: Optional[str] = None,
) -> None:
    """Обрабатывает одно видео и сохраняет временной ряд метрик в CSV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    analyzer = VideoAnalyzer(render_overlay=render_overlay, use_emotions=use_emotions)

    print(f"[+] Старт обработки {os.path.basename(video_path)} (FPS ~ {fps:.1f})")

    rows = []
    event_logger = EventLogger(event_log_path) if event_log_path else None
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_step = max(1, int(sample_every))
    if total_frames > 0:
        max_samples = 300
        sample_step = max(sample_step, max(1, total_frames // max_samples))

    try:
        if total_frames > 0:
            for frame_idx in range(0, total_frames, sample_step):
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
                    continue
                ret, frame = cap.read()
                if not ret:
                    continue

                processed_frame = analyzer.process_frame(frame)
                timestamp = frame_idx / fps if fps else frame_idx
                rows.append(analyzer.snapshot_metrics(timestamp))
                if event_logger:
                    event_logger.update(timestamp, collect_risk_state(analyzer))

                if render_overlay:
                    cv2.imshow("ML_Suicide Offline Preview", processed_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

        while total_frames <= 0:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                processed_frame = analyzer.process_frame(frame)
                timestamp = frame_idx / fps if fps else frame_idx
                rows.append(analyzer.snapshot_metrics(timestamp))
                if event_logger:
                    event_logger.update(timestamp, collect_risk_state(analyzer))

                if render_overlay:
                    cv2.imshow("ML_Suicide Offline Preview", processed_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        print("⏹️  Прервано пользователем.")
                        break

            frame_idx += 1
    finally:
        cap.release()
        analyzer.cleanup()
        if render_overlay:
            cv2.destroyAllWindows()
        if event_logger:
            last_ts = float(rows[-1]["timestamp"]) if rows else None
            event_logger.close(last_ts)

    if not rows:
        print("⚠️  Нет данных для записи (видео пустое?)")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[✓] Готово: {output_csv} ({len(rows)} строк)")


def parse_args():
    parser = argparse.ArgumentParser(description="Офлайн-анализ видео и экспорт в CSV.")
    parser.add_argument(
        "--input_dir",
        default="test_videos",
        help="Каталог с видео (используется, если не указаны --videos)",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Конкретные видеофайлы (пути). Если указаны, обрабатываются только они.",
    )
    parser.add_argument(
        "--output_dir",
        default="video_outputs",
        help="Куда класть CSV отчеты.",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=3,
        help="Сэмплировать каждый N-й кадр (3 ≈ 10 Гц при 30 FPS).",
    )
    parser.add_argument(
        "--emotions",
        action="store_true",
        default=False,
        help="Принудительно включить эмоции даже при офлайн обработке.",
    )
    parser.add_argument(
        "--no_emotions",
        action="store_true",
        help="Принудительно выключить эмоции.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Показывать окно с отрисовкой (медленнее).",
    )
    parser.add_argument(
        "--no_events",
        action="store_true",
        help="Не сохранять лог событий (JSONL).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
        help="Расширения файлов при сканировании каталога.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Отладочный вывод
    if args.videos:
        print(f"[DEBUG] Получено {len(args.videos)} аргументов в --videos")
        for i, v in enumerate(args.videos[:5], 1):  # Показываем первые 5
            print(f"  {i}. {v}")

    if args.videos:
        # Обработка списка видео
        video_list = []
        for v in args.videos:
            if isinstance(v, str):
                # Проверяем, существует ли файл
                if os.path.isfile(v):
                    video_list.append(os.path.abspath(v))
                else:
                    # Может быть путь с пробелами или относительный путь
                    # Пытаемся найти файл
                    abs_path = os.path.abspath(v)
                    if os.path.isfile(abs_path):
                        video_list.append(abs_path)
                    else:
                        print(f"⚠️  Файл не найден: {v}")
        
        # Удаляем дубликаты и сортируем
        video_list = sorted(list(set(video_list)))
        
        if not video_list:
            print("⚠️  Не найдено валидных видеофайлов в --videos.")
            print(f"   Получено аргументов: {len(args.videos)}")
            print(f"   Примеры: {args.videos[:3]}")
            return
    else:
        video_list = discover_videos(args.input_dir, args.extensions)
        if not video_list:
            print("⚠️  В каталоге нет подходящих видео. Укажите --videos.")
            return

    effective_emotions: Optional[bool]
    if args.no_emotions:
        effective_emotions = False
    elif args.emotions:
        effective_emotions = True
    else:
        effective_emotions = None  # оставляем настройку RealtimeAnalyzer

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n📹 Найдено видео для обработки: {len(video_list)}")
    for i, v in enumerate(video_list, 1):
        print(f"  {i}. {os.path.basename(v)}")
    print()

    for video_path in video_list:
        if not os.path.isfile(video_path):
            print(f"⚠️  Пропуск: файл не найден {video_path}")
            continue

        base = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(args.output_dir, f"{base}.csv")

        try:
            event_log_path = None
            if not args.no_events:
                event_log_path = os.path.join(args.output_dir, f"{base}_events.jsonl")

            process_video_file(
                video_path=video_path,
                output_csv=csv_path,
                sample_every=args.sample_every,
                render_overlay=args.preview,
                use_emotions=effective_emotions,
                event_log_path=event_log_path,
            )
        except KeyboardInterrupt:
            print("\n⏹️  Прервано пользователем.")
            break
        except Exception as exc:
            import traceback
            print(f"❌ Ошибка при обработке {video_path}: {exc}")
            print("Детали ошибки:")
            traceback.print_exc()


if __name__ == "__main__":
    main()

