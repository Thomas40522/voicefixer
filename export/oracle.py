import soundfile as sf
import torch
import torchaudio
import onnxruntime as ort
import numpy as np

# ==========================================
# Constants
# ==========================================

MIN_DB = -115.0
MIN_LEVEL_DB = -100.0
MAX_ABS_VALUE = 4.0

# ==========================================
# Mel weight
# ==========================================

def get_mel_weight():
    a = 18.8927416350036
    b = 0.0269863588184314

    x = torch.arange(
        1,
        129,
        dtype=torch.float32
    )

    return a * torch.exp(b * x)

# ==========================================
# Vocoder preprocessing
# ==========================================

def amp_to_db_torch(x):

    min_level = torch.exp(
        torch.tensor(
            MIN_LEVEL_DB / 20.0 * np.log(10.0),
            dtype=x.dtype,
            device=x.device
        )
    )

    return 20.0 * torch.log10(
        torch.maximum(
            x,
            min_level
        )
    )

def normalize_torch(S):

    return torch.clamp(
        (2.0 * MAX_ABS_VALUE)
        * ((S - MIN_DB) / (-MIN_DB))
        - MAX_ABS_VALUE,

        -MAX_ABS_VALUE,
        MAX_ABS_VALUE
    )

def vocoder_preprocess(mel):

    mel_weight = get_mel_weight().to(
        mel.device
    )

    mel = mel / mel_weight.view(
        1, 1, 1, 128
    )

    mel = normalize_torch(
        amp_to_db_torch(
            torch.abs(mel)
        ) - 20.0
    )

    mel = mel[:, 0].transpose(
        1,
        2
    )

    T = mel.shape[2]

    pad_tail = (T % 2) + 4

    padding = torch.full(
        (
            mel.shape[0],
            128,
            pad_tail
        ),
        -4.0,
        dtype=mel.dtype
    )

    mel = torch.cat(
        [mel, padding],
        dim=2
    )

    return mel

# ==========================================
# Load audio
# ==========================================

wav, sr = sf.read(
    "test/original.wav"
)

if wav.ndim > 1:
    wav = wav.mean(axis=1)

wav = torch.tensor(
    wav,
    dtype=torch.float32
)

if sr != 44100:

    resampler = torchaudio.transforms.Resample(
        sr,
        44100
    )

    wav = resampler(
        wav.unsqueeze(0)
    ).squeeze(0)

# ==========================================
# STFT
# ==========================================

stft = torch.stft(
    wav.unsqueeze(0),
    n_fft=2048,
    hop_length=441,
    win_length=2048,
    window=torch.hann_window(2048),
    center=True,
    return_complex=True
)

mag = torch.abs(stft)

mag = mag.unsqueeze(1)

# ==========================================
# Mel Filterbank
# ==========================================

mel_fb = torch.tensor(
    np.load("mel_filterbank.npy"),
    dtype=torch.float32
)

mel = torch.matmul(
    mag.transpose(-1, -2),
    mel_fb
).transpose(-1, -2)

mel = mel.permute(
    0,
    1,
    3,
    2
)

print("Mel shape:", mel.shape)

# ==========================================
# Vocoder preprocessing
# ==========================================

mel = vocoder_preprocess(
    mel
)

print("Vocoder input:", mel.shape)

# ==========================================
# ONNX Vocoder
# ==========================================

session = ort.InferenceSession(
    "voicefixer_vocoder.onnx"
)

audio = session.run(
    None,
    {
        "mel": mel.numpy()
    }
)[0]

print("Audio shape:", audio.shape)

# ==========================================
# Save
# ==========================================

sf.write(
    "test/oracle_onnx.wav",
    audio.squeeze(),
    44100
)