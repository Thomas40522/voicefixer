import torch
import onnxruntime as ort
import numpy as np

from voicefixer.vocoder.base import Vocoder

vocoder = Vocoder(sample_rate=44100)

model = vocoder.model

model.eval()

dummy = torch.randn(
    1,
    128,
    306
)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        "voicefixer_vocoder.onnx",

        input_names=["mel"],
        output_names=["audio"],

        opset_version=17,

        dynamic_axes={
            "mel": {
                2: "time"
            },
            "audio": {
                2: "samples"
            }
        },

        dynamo=False
    )

print("Export complete")

with torch.no_grad():
    torch_out = model(dummy)

session = ort.InferenceSession(
    "voicefixer_vocoder.onnx"
)

onnx_out = session.run(
    None,
    {
        "mel": dummy.numpy()
    }
)[0]

diff = np.abs(
    torch_out.numpy()
    - onnx_out
)

print("Mean:", diff.mean())
print("Max :", diff.max())
print("RMSE:", np.sqrt(np.mean(diff**2)))