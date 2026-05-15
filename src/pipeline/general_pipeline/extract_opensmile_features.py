import os
import platform
import subprocess
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.utils.paths import OPENSMILE_DIR


def _is_supported_smile_binary(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False

    try:
        header = path.read_bytes()[:4]
    except OSError:
        return False

    system = platform.system()
    if system == "Windows":
        return header.startswith(b"MZ")
    if system == "Darwin":
        return header in {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe"}
    return header.startswith(b"\x7fELF")


def find_smilextract_binary(opensmile_dir: str = str(OPENSMILE_DIR)) -> str:
    possible_paths = [
        os.path.join(opensmile_dir, "bin", "SMILExtract.exe"),
        os.path.join(opensmile_dir, "bin", "SMILExtract"),
        os.path.join(opensmile_dir, "build", "progsrc", "smilextract", "SMILExtract"),
        os.path.join(opensmile_dir, "build", "progsrc", "smilextract", "SMILExtract.exe"),
        os.path.join(opensmile_dir, "build", "bin", "SMILExtract"),
        os.path.join(opensmile_dir, "build", "bin", "SMILExtract.exe"),
        "SMILExtract"
    ]
    
    for path in possible_paths:
        candidate = Path(path)
        if candidate.exists() and os.access(candidate, os.X_OK) and _is_supported_smile_binary(candidate):
            return str(candidate.resolve())
    
    raise FileNotFoundError(f"smilextract not found. check paths: {possible_paths}")


def find_egemaps_config(opensmile_dir: str = str(OPENSMILE_DIR)) -> str:
    possible_versions = ["v02", "v01b", "v01a"]
    
    for version in possible_versions:
        config_path = os.path.join(
            opensmile_dir,
            "config",
            "egemaps",
            version,
            f"eGeMAPS{version}.conf"
        )
        
        if os.path.exists(config_path):
            return os.path.abspath(config_path)
    
    raise FileNotFoundError(f"egemaps config not found. checked versions: {possible_versions} in {os.path.join(opensmile_dir, 'config', 'egemaps')}")


def extract_features_opensmile(
    audio_path: str,
    output_path: str,
    smilextract_path: str,
    config_path: str
) -> bool:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            smilextract_path,
            '-C', config_path,
            '-I', audio_path,
            '-O', output_path,
            '-csvoutput', '1',
            '-l', '1',
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if not os.path.exists(output_path):
            print(f"warning: {output_path} not created after processing {audio_path}")
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"error processing {audio_path}:")
        print(f"command: {' '.join(cmd)}")
        print(f"stderr: {e.stderr[:500] if e.stderr else 'no output'}")
        return False
    except Exception as e:
        print(f"unexpected error: {e}")
        return False


def parse_opensmile_csv(csv_path: str) -> dict:
    try:
        if not os.path.exists(csv_path):
            return {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        if len(lines) < 2:
            return {}
        
        features = {}
        attribute_names = []
        data_started = False
        data_line = None
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('%'):
                continue
            
            if line.startswith('@attribute'):
                parts = line.split()
                if len(parts) >= 2:
                    attr_name = parts[1]
                    if attr_name != 'name':
                        attribute_names.append(attr_name)
            
            elif line.startswith('@data'):
                data_started = True
                continue
            
            elif data_started and line:
                data_line = line.split(',')
                break
        
        if not data_line or len(attribute_names) == 0:
            return {}
        
        for i, attr_name in enumerate(attribute_names):
            data_idx = i + 1
            if data_idx < len(data_line):
                try:
                    value_str = data_line[data_idx].strip().strip('"\'')
                    if value_str and value_str != '?' and value_str != 'unknown':
                        value = float(value_str)
                        if not np.isnan(value) and not np.isinf(value):
                            features[attr_name] = value
                except (ValueError, IndexError):
                    continue
        
        return features
        
    except Exception as e:
        print(f"error parsing {csv_path}: {e}")
        return {}


# --- Librosa-based lightweight fallback for environments without OpenSMILE ---
def extract_features_librosa_segment(segment_file: str) -> dict:
    try:
        import soundfile as sf
        import torch
        import torchaudio
    except ImportError:
        raise ImportError(
            "soundfile and torchaudio are required for the lightweight fallback. "
            "Install with `pip install soundfile torchaudio`"
        )

    y, sr = sf.read(segment_file, always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return {}

    waveform = torch.from_numpy(y).unsqueeze(0)
    features = {}

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=int(sr),
        n_mfcc=13,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 40,
            "center": False,
        },
    )
    mfcc = mfcc_transform(waveform).squeeze(0).detach().cpu().numpy()
    for i in range(mfcc.shape[0]):
        coef = mfcc[i]
        features[f"mfcc_{i+1}_mean"] = float(np.mean(coef))
        features[f"mfcc_{i+1}_std"] = float(np.std(coef)) if len(coef) > 1 else 0.0

    # Simple spectral centroid via FFT on the full segment
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / float(sr))
    if spectrum.size and np.sum(spectrum) > 0:
        centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    else:
        centroid = 0.0
    features["spectral_centroid_mean"] = centroid
    features["spectral_centroid_std"] = 0.0

    # zero crossing rate and RMS
    if len(y) > 1:
        zcr = np.mean((y[:-1] * y[1:]) < 0)
    else:
        zcr = 0.0
    features["zero_crossing_rate_mean"] = float(zcr)
    features["rms_mean"] = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0

    # Lightweight fallback does not estimate tempo robustly for short segments.
    features["tempo"] = 0.0

    return features


def batch_extract_features_librosa(
    segments_dir: str,
    output_dir: str,
    file_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> pd.DataFrame:
    segments_path = Path(segments_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_csv_path = output_path / "opensmile_features.csv"
    existing_df = None
    existing_ids: set[str] = set()

    if output_csv_path.exists():
        existing_df = pd.read_csv(output_csv_path)
        existing_ids = set(existing_df["file_id"].unique())

    video_dirs = sorted(d for d in segments_path.iterdir() if d.is_dir())
    if file_ids:
        allowed = {fid.strip() for fid in file_ids}
        video_dirs = [d for d in video_dirs if d.name in allowed]
    if not force:
        video_dirs = [d for d in video_dirs if d.name not in existing_ids]
    if limit is not None:
        video_dirs = video_dirs[:limit]

    if not video_dirs:
        if existing_df is not None:
            print("no new segments to process. using existing csv")
            return existing_df
        print("no segments found for processing")
        return pd.DataFrame()

    all_features = []
    for video_dir in tqdm(video_dirs, desc="extracting features (fallback)"):
        segment_files = sorted([
            f for f in video_dir.iterdir()
            if f.is_file() and f.suffix == ".wav" and (f.stem.startswith("segment_") or f.stem.isdigit())
        ])
        if not segment_files:
            print(f"warning: no segments in {video_dir.name}")
            continue

        for segment_file in segment_files:
            if segment_file.stem.startswith("segment_"):
                segment_num = segment_file.stem.replace("segment_", "")
            else:
                segment_num = segment_file.stem
            segment_id = f"{video_dir.name}_segment_{segment_num}"

            try:
                feats = extract_features_librosa_segment(str(segment_file))
                feats["segment_id"] = segment_id
                feats["file_id"] = video_dir.name
                all_features.append(feats)
            except ImportError as e:
                print(f"error: {e}")
                return pd.DataFrame()
            except Exception as e:
                print(f"warning: failed to extract features from {segment_file}: {e}")
                continue

    if not all_features:
        print("warning: failed to extract features from any segment")
        if existing_df is not None:
            print("returning existing csv unchanged")
        return existing_df if existing_df is not None else pd.DataFrame()

    print(f"successfully extracted features from {len(all_features)} segments (fallback)")

    features_df = pd.DataFrame(all_features)

    if existing_df is not None:
        new_file_ids = set(features_df["file_id"].unique())
        overlap = new_file_ids & existing_ids
        if overlap:
            print(f"warning: files {overlap} will be overwritten")
            existing_df = existing_df[~existing_df["file_id"].isin(new_file_ids)]
        features_df = pd.concat([existing_df, features_df], ignore_index=True)

    return features_df


def batch_extract_features(
    segments_dir: str,
    output_dir: str,
    opensmile_dir: str = str(OPENSMILE_DIR),
    file_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> pd.DataFrame:
    segments_path = Path(segments_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_csv_path = output_path / "opensmile_features.csv"
    existing_df = None
    existing_ids: set[str] = set()

    if output_csv_path.exists():
        existing_df = pd.read_csv(output_csv_path)
        existing_ids = set(existing_df["file_id"].unique())

    try:
        smilextract_path = find_smilextract_binary(opensmile_dir)
        config_path = find_egemaps_config(opensmile_dir)
        print(f"using smilextract: {smilextract_path}")
        print(f"using config: {config_path}")
        using_smile = True
    except FileNotFoundError:
        print(f"warning: SMILExtract not found in {opensmile_dir}. Falling back to lightweight acoustic features.")
        using_smile = False

    if not segments_path.exists():
        print(f"warning: segments directory not found: {segments_dir}")
        if existing_df is not None:
            return existing_df
        return pd.DataFrame()

    video_dirs = sorted(d for d in segments_path.iterdir() if d.is_dir())
    if file_ids:
        allowed = {fid.strip() for fid in file_ids}
        video_dirs = [d for d in video_dirs if d.name in allowed]
    if not force:
        video_dirs = [d for d in video_dirs if d.name not in existing_ids]
    if limit is not None:
        video_dirs = video_dirs[:limit]

    if not video_dirs:
        if existing_df is not None:
            print("no new segments to process. using existing csv")
            return existing_df
        print("no segments found for processing")
        return pd.DataFrame()

    # If SMILExtract is not available, use librosa fallback
    if not using_smile:
        return batch_extract_features_librosa(segments_dir, output_dir, file_ids=file_ids, limit=limit, force=force)

    all_features = []
    total_segments = sum(
        len([f for f in d.iterdir() if f.is_file() and f.suffix == ".wav" and (f.stem.startswith("segment_") or f.stem.isdigit())])
        for d in video_dirs
    )
    print(f"processing {len(video_dirs)} videos, total {total_segments} segments...")
    
    for video_dir in tqdm(video_dirs, desc="extracting features"):
        segment_files = sorted([
            f for f in video_dir.iterdir()
            if f.is_file() and f.suffix == ".wav" and (f.stem.startswith("segment_") or f.stem.isdigit())
        ])
        if not segment_files:
            print(f"warning: no segments in {video_dir.name}")
            continue

        for segment_file in segment_files:
            if segment_file.stem.startswith("segment_"):
                segment_num = segment_file.stem.replace("segment_", "")
            else:
                segment_num = segment_file.stem
            segment_id = f"{video_dir.name}_segment_{segment_num}"
            output_csv = output_path / f"{segment_id}.csv"

            if extract_features_opensmile(
                str(segment_file),
                str(output_csv),
                smilextract_path,
                config_path,
            ):
                features = parse_opensmile_csv(str(output_csv))
                if features:
                    features["segment_id"] = segment_id
                    features["file_id"] = video_dir.name
                    all_features.append(features)
                else:
                    print(f"warning: failed to parse features from {output_csv}")

    if not all_features:
        print("warning: failed to extract features from any segment")
        if existing_df is not None:
            print("returning existing csv unchanged")
        return existing_df if existing_df is not None else pd.DataFrame()
    
    print(f"successfully extracted features from {len(all_features)} segments")

    features_df = pd.DataFrame(all_features)

    if existing_df is not None:
        new_file_ids = set(features_df["file_id"].unique())
        overlap = new_file_ids & existing_ids
        if overlap:
            print(f"warning: files {overlap} will be overwritten")
            existing_df = existing_df[~existing_df["file_id"].isin(new_file_ids)]
        features_df = pd.concat([existing_df, features_df], ignore_index=True)

    return features_df


def main():
    parser = argparse.ArgumentParser(description='extract acoustic features with opensmile')
    parser.add_argument('--segments-dir', type=str, default='data/segments')
    parser.add_argument('--output-dir', type=str, default='data/features')
    parser.add_argument('--opensmile-dir', type=str, default='opensmile')
    parser.add_argument('--output-csv', type=str, default=None)
    parser.add_argument('--file-ids', nargs='+', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    
    args = parser.parse_args()
    
    features_df = batch_extract_features(
        args.segments_dir,
        args.output_dir,
        args.opensmile_dir,
        file_ids=args.file_ids,
        limit=args.limit,
        force=args.force
    )
    
    if not features_df.empty:
        if args.output_csv is None:
            output_csv_path = Path(args.output_dir) / "opensmile_features.csv"
        else:
            output_csv_path = Path(args.output_csv)
        
        os.makedirs(output_csv_path.parent, exist_ok=True)
        features_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"features saved to {output_csv_path}")
        print(f"total segments: {len(features_df)}")
        print(f"total features: {len(features_df.columns)}")
        print(f"unique videos: {features_df['file_id'].nunique()}")
    else:
        print("failed to extract features")


if __name__ == '__main__':
    main()
