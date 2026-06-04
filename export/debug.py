import onnx
import torch
from voicefixer.restorer.model import VoiceFixer
from voicefixer.restorer.wrapper import GeneratorWrapper
import onnxruntime as ort
import numpy as np
import os
from voicefixer.tools.mel_scale import MelScale


# ##############
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

# unet = generator.unet

# unet.eval()

# ##############


# dummy = torch.randn(
#     1, 2, 512, 128
# )  # because unet_in = cat([to_log(mel_orig), x], dim=1)

# torch.onnx.export(
#     unet,
#     dummy,
#     "unet.onnx",
#     input_names=["input"],

#     output_names=["mel"],
#     dynamic_axes={
#         "input": {2: "time"},
#         "mel": {2: "time"},
#     },

#     opset_version=17,
#     dynamo=False
# )

# session = ort.InferenceSession("unet.onnx")

# out = session.run(
#     None,
#     {"input": np.random.randn(1,2,301,128).astype(np.float32)}
# )

# for i,o in enumerate(out):
#     print(i, o.shape)

# # print(out)

mel_layer = MelScale(
    n_mels=128,
    sample_rate=44100,
    n_stft=1025
)

fb = mel_layer.fb

print(fb.shape)

np.save(
    "mel_filterbank.npy",
    fb.numpy()
)

