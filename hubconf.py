import torch
import os
from typing import Optional
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

dependencies = ["torch", "numpy", "geffnet"]
def _load_state_dict(local_file_path: Optional[str] = None):
    if local_file_path is not None and os.path.exists(local_file_path):
        # Load state_dict from local file
        state_dict = torch.load(local_file_path, map_location=torch.device("cpu"))
    else:
        # Load state_dict from the default URL
        file_name = "dsine.pt"
        url = f"https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
        state_dict = torch.hub.load_state_dict_from_url(url, file_name=file_name, map_location=torch.device("cpu"))

    return state_dict['model']

class Predictor:
    def __init__(self, model) -> None:
        from models.dsine import DSINE
        self.device = torch.device('cuda')
        self.model = model
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def infer_cv2(self, image):
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.infer_pil(image)
    
    def infer_pil(self, img, intrins=None):
        import utils.utils as utils
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        _, _, orig_H, orig_W = img.shape

        # zero-pad the input image so that both the width and height are multiples of 32
        l, r, t, b = utils.pad_input(orig_H, orig_W)
        img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)
        img = self.transform(img)

        if intrins is None:
            intrins = utils.get_intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=self.device).unsqueeze(0)
        
        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t

        with torch.no_grad():
            pred_norm = self.model(img, intrins=intrins)[-1]
            pred_norm = pred_norm[:, :, t:t+orig_H, l:l+orig_W]
        
        # pred_norm_np = pred_norm.cpu().detach().numpy()[0,:,:,:].transpose(1, 2, 0) # (H, W, 3)
        return pred_norm

def DSINE(local_file_path: Optional[str] = None):
    from models import dsine

    state_dict = _load_state_dict(local_file_path)
    model = dsine.DSINE()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(torch.device("cuda"))
    model.pixel_coords = model.pixel_coords.to(torch.device("cuda"))

    return Predictor(model)


def _test_run():
    import argparse
    import torch.nn.functional as F
    import numpy as np

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output image file")
    parser.add_argument("--remote", action="store_true", help="use remote repo")
    parser.add_argument("--reload", action="store_true", help="reload remote repo")
    parser.add_argument("--pil", action="store_true", help="use PIL instead of OpenCV")
    args = parser.parse_args()

    if not args.remote:
        predictor = torch.hub.load(".", "DSINE", local_file_path='./checkpoints/dsine.pt',
                                source="local", trust_repo=True)
    else:
        predictor = torch.hub.load(".", "DSINE",
                                source="local", trust_repo=True)
        
    if args.pil:
        import PIL
        import torchvision.transforms.functional as TF
        
        image = PIL.Image.open(args.input).convert("RGB")
        h, w = image.height, image.width
        with torch.inference_mode():
            normal = predictor.infer_pil(image)[0] # (H, W, 3)
            normal = (normal + 1) / 2

        normal = TF.to_pil_image(normal.cpu())
        normal.save(args.output)
        
    else:
        import cv2
        image = cv2.imread(args.input, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        with torch.inference_mode():
            normal = predictor.infer_cv2(image)[0] # (H, W, 3)
            normal = (normal + 1) / 2

        normal = (normal * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, normal)


if __name__ == "__main__":
    _test_run()