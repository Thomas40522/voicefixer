import torch

class GeneratorWrapper(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, sp, mel):
        out = self.generator(sp, mel)

        # return (
        #     out["x"],
        #     out["unet_out"]
        # )
        return out["mel"]