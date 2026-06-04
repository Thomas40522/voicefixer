import soundfile as sf
import numpy as np
import torch
import onnxruntime as ort
import torchaudio
import os

wav, sr = sf.read("test/original.wav")

if wav.ndim > 1:
    wav = wav.mean(axis=1)

wav = torch.tensor(
    wav,
    dtype=torch.float32
)

if sr != 44100:
    resampler = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=44100
    )

    wav = resampler(
        wav.unsqueeze(0)
    ).squeeze(0)

    sr = 44100

wav = wav.unsqueeze(0)

stft = torch.stft(
    wav,
    n_fft=2048,
    hop_length=441,
    win_length=2048,
    window=torch.hann_window(2048),
    center=True,
    return_complex=True
)

mag = torch.abs(stft)

mag = mag.unsqueeze(1)

mel_filterbank = torch.tensor(
    np.load("mel_filterbank.npy"),
    dtype=torch.float32
)

mel = torch.matmul(
    mag.transpose(-1, -2),
    mel_filterbank
).transpose(-1, -2)

mel = mel.permute(0, 1, 3, 2)

print("preprocessing complete, mel shape:", mel.shape)

##### Model Inference #####

generator_session = ort.InferenceSession(
    "voicefixer_generator.onnx"
)

outputs = generator_session.run(
    None,
    {
        "mel": mel.numpy()
    }
)

onnx_mel = outputs[0]

print("Inference completed, ONNX mel shape:", onnx_mel.shape)

###### Reconstruction ######

MIN_DB = -115.0
MIN_LEVEL_DB = -100.0
MAX_ABS_VALUE = 4.0


def get_mel_weight():
    a = 18.8927416350036
    b = 0.0269863588184314

    x = torch.arange(
        1,
        129,
        dtype=torch.float32
    )

    return a * torch.exp(b * x)

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
        dtype=mel.dtype,
        device=mel.device
    )

    mel = torch.cat(
        [mel, padding],
        dim=2
    )

    return mel

vocoder_session = ort.InferenceSession(
    "voicefixer_vocoder.onnx"
)


mel_out = np.clip(
    onnx_mel,
    -100,
    5
)

mel_out = 10 ** mel_out

mel_out = torch.tensor(
    mel_out,
    dtype=torch.float32
)

vocoder_input = vocoder_preprocess(
    mel_out
)

audio = vocoder_session.run(
    None,
    {
        "mel": vocoder_input.numpy()
    }
)[0]

sf.write(
    "test/onnx_output.wav",
    audio.squeeze(),
    44100
)

print("Reconstruction complete, output saved to test/onnx_output.wav")