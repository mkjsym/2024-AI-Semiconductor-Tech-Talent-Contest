import cv2
import numpy as np

from typing import Any, Dict, List, Tuple

class YOLOPreProcessor:
    @staticmethod
    def __call__(image: np.ndarray, new_shape: Tuple[int, int] = (640, 640), tensor_type: str = "uint8") -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        img, preproc_params = letterbox(image, new_shape)
        img = img.transpose([2, 0, 1])[::-1]
        contexts = {"ratio": preproc_params[0], "pad": preproc_params[1]}

        if tensor_type == "uint8":
            input_data = np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.uint8)
        else:
            # Normalize for float32
            input_data = np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.float32) / 255.0
        
        return input_data, contexts

def letterbox(image: np.ndarray, new_shape: Tuple[int,int], color=(114,114,114), scaleup:bool = True):
    h, w = image.shape[:2]
    if h == new_shape[0] and w == new_shape[1]:
        return image, (1.0, (0, 0))
    
    ratio = min(new_shape[0]/h, new_shape[1]/w)
    if not scaleup:
        ratio = min(ratio, 1.0)

    new_unpad = int(round(ratio * w)), int(round(ratio * h))
    dw, dh = (new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1])
    dw /= 2
    dh /= 2

    if ratio != 1.0:
        interpolation = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
        image = cv2.resize(image, new_unpad, interpolation=interpolation)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, (ratio, (dw, dh))


def letterbox(image, new_shape, color=(114,114,114)):
    h, w = image.shape[:2]
    if h == new_shape[0] and w == new_shape[1]:
        return image, (1.0, (0, 0))
    
    ratio = min(new_shape[0]/h, new_shape[1]/w)

    new_unpad = int(round(ratio * w)), int(round(ratio * h))
    dw = (new_shape[1] - new_unpad[0])/2
    dh = (new_shape[0] - new_unpad[1])/2
    
    if ratio != 1.0:
        interpolation = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
        image = cv2.resize(image, new_unpad, interpolation=interpolation)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, (ratio, (dw, dh))