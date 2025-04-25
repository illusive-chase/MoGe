from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from rfstudio.engine.task import Task
from rfstudio.graphics import DepthImages
from rfstudio.io import dump_float32_image, load_float32_image
from rfstudio.utils.colormap import IntensityColorMap

from moge.model.v1 import MoGeModel


@dataclass
class Script(Task):

    input: Path = ...
    output: Path = ...
    masked: bool = True
    colormap: Literal['viridis', 'plasma', 'inferno', 'magma', 'cividis'] = 'magma'

    def run(self) -> None:

        assert self.output.suffix in ['.exr', '.png', '.jpg']

        # Load the model from huggingface hub (or load from local).
        model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)

        # Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
        input_image = load_float32_image(self.input).permute(2, 0, 1).to(self.device)

        # Infer 
        output = model.infer(input_image)
        # `output` has keys "points", "depth", "mask" and "intrinsics",
        # The maps are in the same size as the input image.
        # {
        #     "points": (H, W, 3),    # scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
        #     "depth": (H, W),        # scale-invariant depth map
        #     "mask": (H, W),         # a binary mask for valid pixels.
        #     "intrinsics": (3, 3),   # normalized camera intrinsics
        # }
        # For more usage details, see the `MoGeModel.infer` docstring.

        depth_map = output["depth"].unsqueeze(-1)
        mask = output["mask"].unsqueeze(-1)
        if not self.masked:
            mask = torch.ones_like(mask)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.suffix in ['.exr']:
            dump_float32_image(self.output, depth_map * mask)
        else:
            vis_depth = DepthImages([
                torch.cat((depth_map.nan_to_num(0., 0., 0.), mask), dim=-1)
            ]).visualize(IntensityColorMap(style=self.colormap))
            dump_float32_image(self.output, vis_depth.item())

if __name__ == '__main__':
    Script(cuda=0).run()

