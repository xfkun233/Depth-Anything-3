# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import imageio
import numpy as np

from depth_anything_3.specs import Prediction


def export_to_depth_raw(
    prediction: Prediction,
    export_dir: str,
):
    """
    Export raw depth maps as grayscale images (0-255 normalized).
    Saves depth maps without colorization to depth_raw/ directory.
    """
    if prediction.depth is None:
        raise ValueError("prediction.depth is required but not available")

    os.makedirs(os.path.join(export_dir, "depth_raw"), exist_ok=True)
    
    for idx in range(prediction.depth.shape[0]):
        depth = prediction.depth[idx]
        
        # Normalize depth to 0-255 range for visualization
        valid_mask = depth > 0
        if valid_mask.any():
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            
            # Avoid division by zero
            if depth_max > depth_min:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth)
            
            # Convert to 0-255 grayscale (inverted: closer = brighter)
            depth_gray = (255 * (1 - depth_normalized)).astype(np.uint8)
            depth_gray[~valid_mask] = 0  # Invalid pixels as black
        else:
            depth_gray = np.zeros_like(depth, dtype=np.uint8)
        
        save_path = os.path.join(export_dir, f"depth_raw/{idx:04d}.png")
        imageio.imwrite(save_path, depth_gray)
