"""
ml/audio_model.py

Lightweight CNN audio classifier that detects synthetic/cloned voice artefacts
from log-mel spectrograms.  Trained on the FakeAVCeleb RealVideo-FakeAudio and
FakeVideo-FakeAudio categories which use SV2TTS (voice cloning) / Wav2Lip.

Input:  (B, 1, n_mels, T) log-mel spectrogram
Output: (B, num_classes)  4-class logits (same class space as video model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class AudioCNN(nn.Module):
    """
    Lightweight ResNet-inspired CNN for log-mel spectrograms.
    Architecture optimised to be fast even on CPU.
    """

    def __init__(self, num_classes=4, in_channels=1):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Residual stages
        self.layer1 = self._make_layer(32,  64, n=2, stride=1)
        self.layer2 = self._make_layer(64, 128, n=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n=2, stride=2)
        # Global average pool → classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n - 1):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: (B, 1, n_mels, T)"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).flatten(1)          # (B, 256)
        x = self.dropout(x)
        return self.classifier(x)           # (B, num_classes)


def build_audio_model(num_classes=4):
    return AudioCNN(num_classes=num_classes)
