import os
import glob
import pandas as pd
import numpy as np
import cv2
from moviepy import VideoFileClip
import argparse
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
import shutil

def extract_frames_and_faces(video_path, out_frame_dir, mtcnn, fps_extract=5):
    # Extract frames using cv2 and crop face
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0 or np.isnan(orig_fps):
        orig_fps = 25
        
    frame_interval = max(1, int(orig_fps / fps_extract))
    
    frame_idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            boxes, probs = mtcnn.detect(frame_rgb)
            if boxes is not None and len(boxes) > 0:
                # Get the most probable face (first box)
                box = boxes[0]
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Expand bounding box slightly
                h, w = frame_rgb.shape[:2]
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    face_resized = cv2.resize(face, (224, 224))
                    out_path = os.path.join(out_frame_dir, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(out_path, face_resized)
                    saved_count += 1
        frame_idx += 1
        
    cap.release()
    return saved_count > 0

def extract_audio(video_path, out_audio_path):
    # Use moviepy to extract audio and save as WAV (16kHz, mono)
    try:
        video = VideoFileClip(video_path)
        if video.audio is not None:
            # Note: moviepy write_audiofile supports fps and nbytes
            # For 16kHz mono, fps=16000, but moviepy might not support all conversions directly.
            # We will use ffmpeg_params to force 16kHz mono
            video.audio.write_audiofile(
                out_audio_path,
                fps=16000,
                nbytes=2,
                codec='pcm_s16le',
                ffmpeg_params=["-ac", "1"],
                logger=None
            )
            video.close()
            return True
        else:
            video.close()
            return False
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to FakeAVCeleb directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save frames, audio, and manifest")
    parser.add_argument("--subset", type=int, default=None, help="Process only a subset of videos for testing")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    audio_dir = output_dir / "audio"
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    categories = {
        "RealVideo-RealAudio": {"label": "real-real", "fake_video": 0, "fake_audio": 0},
        "FakeVideo-RealAudio": {"label": "fake_video-real_audio", "fake_video": 1, "fake_audio": 0},
        "RealVideo-FakeAudio": {"label": "real_video-fake_audio", "fake_video": 0, "fake_audio": 1},
        "FakeVideo-FakeAudio": {"label": "fake_video-fake_audio", "fake_video": 1, "fake_audio": 1}
    }

    print("Scanning dataset directory...")
    records = []
    
    # FakeAVCeleb structure: category/ethnicity/gender/identity/video.mp4
    for cat, meta in categories.items():
        cat_path = dataset_dir / cat
        if not cat_path.exists():
            continue
        
        # Find all mp4 files
        for mp4_file in cat_path.rglob("*.mp4"):
            identity = mp4_file.parent.name
            records.append({
                "original_filepath": str(mp4_file.absolute()),
                "category": cat,
                "label": meta["label"],
                "identity": identity,
                "has_fake_video": meta["fake_video"],
                "has_fake_audio": meta["fake_audio"]
            })

    df = pd.DataFrame(records)
    print(f"Found {len(df)} total videos.")

    if args.subset:
        df = df.sample(n=min(args.subset, len(df)), random_state=42).reset_index(drop=True)
        print(f"Using subset of {len(df)} videos.")

    # Split by identity
    unique_ids = df['identity'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_ids)
    
    n_total = len(unique_ids)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train+n_val])
    
    def assign_split(ident):
        if ident in train_ids: return 'train'
        elif ident in val_ids: return 'val'
        else: return 'test'

    df['split'] = df['identity'].apply(assign_split)

    # Initialize MTCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing MTCNN on {device}...")
    mtcnn = MTCNN(keep_all=False, device=device)

    processed_records = []

    print("Processing videos...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        orig_path = row["original_filepath"]
        vid_id = f"{row['category']}_{row['identity']}_{Path(orig_path).stem}"
        
        vid_frames_dir = frames_dir / vid_id
        vid_frames_dir.mkdir(exist_ok=True)
        
        audio_path = audio_dir / f"{vid_id}.wav"
        
        # Extract audio
        audio_success = extract_audio(orig_path, str(audio_path))
        
        # Extract frames
        frames_success = extract_frames_and_faces(orig_path, str(vid_frames_dir), mtcnn)
        
        if not audio_success or not frames_success:
            print(f"Failed to process {vid_id}, skipping.")
            if vid_frames_dir.exists():
                shutil.rmtree(vid_frames_dir)
            if audio_path.exists():
                audio_path.unlink()
            continue
            
        row_dict = row.to_dict()
        row_dict["frames_dir"] = str(vid_frames_dir.absolute())
        row_dict["audio_path"] = str(audio_path.absolute())
        processed_records.append(row_dict)

    if not processed_records:
        print("No videos were successfully processed.")
        return

    out_df = pd.DataFrame(processed_records)
    manifest_path = output_dir / "manifest.csv"
    out_df.to_csv(manifest_path, index=False)
    print(f"Successfully processed {len(out_df)} videos.")
    print(f"Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
