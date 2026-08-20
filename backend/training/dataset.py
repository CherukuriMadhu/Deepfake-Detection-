import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from audiomentations import Compose as AudioCompose, AddGaussianNoise, TimeStretch, PitchShift

class FakeAVCelebDataset(Dataset):
    def __init__(self, manifest_path, split='train', num_frames=32, audio_target_length=16000*3, transform=None, audio_transform=None):
        self.manifest = pd.read_csv(manifest_path)
        self.manifest = self.manifest[self.manifest['split'] == split].reset_index(drop=True)
        self.num_frames = num_frames
        self.audio_target_length = audio_target_length
        self.transform = transform
        self.audio_transform = audio_transform
        
        # 4-class labels: 
        # 0: real-real
        # 1: fake_video-real_audio
        # 2: real_video-fake_audio
        # 3: fake_video-fake_audio
        self.label_map = {
            "real-real": 0,
            "fake_video-real_audio": 1,
            "real_video-fake_audio": 2,
            "fake_video-fake_audio": 3
        }
        
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        frames_dir = row['frames_dir']
        audio_path = row['audio_path']
        label_str = row['label']
        label = self.label_map[label_str]
        
        # --- 1. Load Video Frames ---
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        
        if len(frame_files) == 0:
            # Fallback if no frames were extracted
            frames = torch.zeros(self.num_frames, 3, 224, 224)
        else:
            if len(frame_files) >= self.num_frames:
                indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
            else:
                indices = np.pad(np.arange(len(frame_files)), 
                               (0, self.num_frames - len(frame_files)), 
                               mode='wrap')
                
            frames = []
            for i in indices:
                img_path = frame_files[i]
                img = cv2.imread(img_path)
                if img is None:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if self.transform:
                    augmented = self.transform(image=img)
                    img = augmented['image']
                else:
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                frames.append(img)
            
            frames = torch.stack(frames)
            
        # --- 2. Load Audio ---
        if os.path.exists(audio_path):
            try:
                waveform, sample_rate = sf.read(audio_path)
                # Ensure mono
                if len(waveform.shape) > 1:
                    waveform = waveform.mean(axis=1)
            except Exception as e:
                print(f"Failed to load {audio_path}: {e}")
                waveform = np.zeros(self.audio_target_length)
                sample_rate = 16000
            
            # Pad or truncate to target length
            if len(waveform) < self.audio_target_length:
                pad_len = self.audio_target_length - len(waveform)
                waveform = np.pad(waveform, (0, pad_len), mode='constant')
            else:
                if self.transform and len(waveform) > self.audio_target_length:
                    start = np.random.randint(0, len(waveform) - self.audio_target_length)
                    waveform = waveform[start:start + self.audio_target_length]
                else:
                    waveform = waveform[:self.audio_target_length]
                    
            if self.audio_transform:
                waveform = self.audio_transform(samples=waveform, sample_rate=16000)
                
            waveform = torch.from_numpy(waveform).unsqueeze(0).float()
            
            # Compute Log-Mel Spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=64, n_fft=1024, hop_length=256
            )
            mel_spec = mel_transform(waveform)
            log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        else:
            # Fallback if audio missing
            log_mel_spec = torch.zeros(1, 64, 188)
            
        return frames, log_mel_spec, torch.tensor(label, dtype=torch.long)

def get_train_transforms():
    return A.Compose([
        A.ImageCompression(quality_range=(60, 100), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
def get_audio_transforms():
    return AudioCompose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
    ])

if __name__ == "__main__":
    # Sanity check
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    
    dataset = FakeAVCelebDataset(
        manifest_path=args.manifest,
        split=args.split,
        transform=get_train_transforms(),
        audio_transform=get_audio_transforms()
    )
    
    print(f"Dataset length for split '{args.split}': {len(dataset)}")
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        frames, log_mel_spec, labels = next(iter(loader))
        
        print(f"Batch frames shape: {frames.shape}") 
        print(f"Batch audio shape: {log_mel_spec.shape}") 
        print(f"Batch labels: {labels}")
