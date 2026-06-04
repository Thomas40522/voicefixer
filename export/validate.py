import soundfile as sf
import numpy as np
import torch
import onnxruntime as ort
import torchaudio
# from voicefixer.vocoder.base import Vocoder


# from voicefixer.tools.mel_scale import MelScale
# from voicefixer.restorer.model import VoiceFixer
import os

####### model

# model = VoiceFixer(channels=2, sample_rate=44100)

# ckpt = os.path.join(os.path.expanduser("~"), ".cache/voicefixer/analysis_module/checkpoints/vf.ckpt")

# saved_state_dict = torch.load(
#     ckpt
# )

# model_state_dict = model.state_dict()

# new_state_dict = {
#     k: v
#     for k, v in saved_state_dict.items()
#     if k in model_state_dict
# }

# model_state_dict.update(new_state_dict)

# model.load_state_dict(
#     model_state_dict,
#     strict=False
# )

# model.eval()

# generator = model.generator

# generator.eval()

#######


#### Audio Preprocessing #####

wav, sr = sf.read("test/original.wav")

assert sr == 44100

if wav.ndim > 1:
    wav = wav.mean(axis=1)

wav = torch.tensor(
    wav,
    dtype=torch.float32
)

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

# mel_layer = MelScale(
#     n_mels=128,
#     sample_rate=44100,
#     n_stft=1025
# )

mel_filterbank = torch.tensor(
    np.load("mel_filterbank.npy"),
    dtype=torch.float32
)

mel = torch.matmul(
    mag.transpose(-1, -2),
    mel_filterbank
).transpose(-1, -2)

mel = mel.permute(0, 1, 3, 2)


# mel = mel_layer(
#     mag
# ).permute(0,1,3,2)

print("preprocessing complete, mel shape:", mel.shape)

##### Model Inference #####


# with torch.no_grad():
#     out = generator(
#         torch.zeros(1,1,1025,301),
#         mel
#     )

# print(out["mel"].shape)

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

# torch_mel = out["mel"].numpy()

# print("Torch shape:", torch_mel.shape)
# print("ONNX shape :", onnx_mel.shape)

# print(
#     "Mean abs error:",
#     np.mean(np.abs(torch_mel - onnx_mel))
# )

# print(
#     "Max abs error:",
#     np.max(np.abs(torch_mel - onnx_mel))
# )

# print(
#     "RMSE:",
#     np.sqrt(np.mean((torch_mel - onnx_mel) ** 2))
# )


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


def reconstruct(mel_np, outfile):

    mel_np = np.clip(
        mel_np,
        a_min=-100,
        a_max=5
    )

    mel_np = 10 ** mel_np

    mel = torch.tensor(
        mel_np,
        dtype=torch.float32
    )

    with torch.no_grad():
        wav = vocoder(
            mel,
            cuda=False
        )

    sf.write(
        outfile,
        wav.squeeze().cpu().numpy(),
        44100
    )

# vocoder = Vocoder(sample_rate=44100)



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



# reconstruct(
#     out["mel"].numpy(),
#     "test/torch_output.wav"
# )

# reconstruct(
#     outputs[0],
#     "test/onnx_output.wav"
# )


