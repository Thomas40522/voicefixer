import numpy as np

mel = np.load("mel_filterbank.npy")

print(mel.shape)
print(mel.dtype)

with open("mel_filterbank_data.h", "w") as f:

    rows, cols = mel.shape

    f.write("#pragma once\n\n")

    f.write(
        f"constexpr int MEL_ROWS = {rows};\n"
    )

    f.write(
        f"constexpr int MEL_COLS = {cols};\n\n"
    )

    f.write(
        "constexpr float MEL_FILTERBANK[] = {\n"
    )

    flat = mel.flatten()

    for value in flat:
        f.write(
            f"{float(value)}f,\n"
        )

    f.write("};\n")