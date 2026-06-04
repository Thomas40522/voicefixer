from voicefixer import VoiceFixer, Vocoder

voicefixer = VoiceFixer()

voicefixer.restore(
    input="test/test/original.wav",
    output="test/test/restored.wav",
    cuda=False,
    mode=0
)

vocoder = Vocoder(sample_rate=44100)

vocoder.oracle(
    fpath="test/test/original.wav",
    out_path="test/test/oracle.wav",
    cuda=False,
)

print("done")