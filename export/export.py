from pathlib import Path
import sys
# Add the parent directory of the current file to the Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from voicefixer.restorer.model import VoiceFixer
from voicefixer.restorer.wrapper import GeneratorWrapper
import onnxruntime as ort
import numpy as np
import os

model = VoiceFixer(channels=2, sample_rate=44100)

ckpt = os.path.join(os.path.expanduser("~"), ".cache/voicefixer/analysis_module/checkpoints/vf.ckpt")

saved_state_dict = torch.load(
    ckpt
)

model_state_dict = model.state_dict()

new_state_dict = {
    k: v
    for k, v in saved_state_dict.items()
    if k in model_state_dict
}

model_state_dict.update(new_state_dict)

model.load_state_dict(
    model_state_dict,
    strict=False
)

model.eval()

generator = model.generator

generator.eval()

chunk_size = 512


sp = torch.randn(1, 1, chunk_size, 1025)

mel = torch.randn(1, 1, chunk_size, 128)

wrapper = GeneratorWrapper(generator)
wrapper.eval()


def export_model():
    torch.onnx.export(
        wrapper,
        (sp, mel),

        "voicefixer_generator.onnx",

        input_names=["sp", "mel"],
        output_names=["mel_out"],

        opset_version=17,
       
        dynamic_axes={
            "sp": {2: "time"},
            "mel": {2: "time"},
            "mel_out": {2: "time"},
            # "unet_out": {2: "time"},
        },

        dynamo=False

    )
    
# export_model()

sp_ex = torch.randn(1, 1, 301, 1025)

mel_ex = torch.randn(1, 1, 301, 128)


with torch.no_grad():
    torch_output = wrapper(sp_ex, mel_ex)
# print("Torch output shapes:")
# # print("x: ", torch_output[0].shape)
# print("mel: ", torch_output[0].shape)


session = ort.InferenceSession(
    "voicefixer_generator.onnx"
)

for inp in session.get_inputs():
    print(inp.name)
    print(inp.shape)

for out in session.get_outputs():
    print(out.name)
    print(out.shape)


onnx_output = session.run(
    None,
    {
        "mel": mel_ex.numpy()
    }
)

# print("ONNX output shapes:")
# print("x: ", onnx_output[0].shape)

diff = np.mean(np.abs(torch_output.numpy() - onnx_output[0]))
print("mel_out: " + str(diff))

# diff = np.mean(np.abs(torch_output[1].numpy() - onnx_output[1]))
# print("unet_out: " + str(diff))

# diff = np.mean(np.abs(torch_output["lstm_out"].numpy() - onnx_output[1]))
# print("lstm_out: " + str(diff))

# diff = np.mean(np.abs(torch_output["unet_out"].numpy() - onnx_output[2]))
# print("unet_out: " + str(diff))

# diff = np.mean(np.abs(torch_output["noisy"].numpy() - onnx_output[3]))
# print("noisy: " + str(diff))

# diff = np.mean(np.abs(torch_output["clean"].numpy() - onnx_output[4]))
# print("clean: " + str(diff))


# print("Difference:", difference)