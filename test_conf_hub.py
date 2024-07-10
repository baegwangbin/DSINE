import torch
import cv2
import numpy as np
import sys

normal_predictor = torch.hub.load("pierremerriaux-leddartech/DSINE", "DSINE", trust_repo=True, source='github')

image = cv2.imread('projects/dsine/samples/img/office_01.png', cv2.IMREAD_COLOR)
h, w = image.shape[:2]

# Use the model to infer the normal map from the input image
with torch.inference_mode():
    normal = normal_predictor.infer_cv2(image)[0]  # Output shape: (H, W, 3)
    normal = (normal + 1) / 2  # Convert values to the range [0, 1]

# Convert the normal map to a displayable format
normal = (normal * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)

# Save the output normal map to a file
cv2.imwrite('projects/dsine/samples/img/office_01_result.png', normal)