"""Cosmos3's image postprocess: the PNG holds the decoder's first frame, whichever
rank the decoder hands it in."""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from mstar.model.cosmos3.cosmos3_model import Cosmos3Model


@pytest.mark.parametrize("layout", ["[1, 3, 1, H, W]", "[1, 3, H, W]", "[3, H, W]"])
def test_the_png_is_the_decoders_frame(layout) -> None:
    image = torch.randint(0, 256, (3, 24, 40), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    output = {"[1, 3, 1, H, W]": image[None, :, None], "[1, 3, H, W]": image[None], "[3, H, W]": image}[layout]
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    with Image.open(io.BytesIO(model.postprocess(output, "image"))) as png:
        assert png.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(png), image.permute(1, 2, 0).numpy())
