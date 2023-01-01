import cv2
import numpy


def resize_image(
    src,
    max_size,
    up_interpolation=cv2.INTER_LANCZOS4,
    down_interpolation=cv2.INTER_AREA,
):
    h, w = src.shape[:2]
    if w > h:
        if w > max_size:
            interpolation = down_interpolation
        else:
            interpolation = up_interpolation

        h = int(max_size * h / (float(w)))
        w = max_size

    else:
        if h > max_size:
            interpolation = down_interpolation
        else:
            interpolation = up_interpolation

        w = int(max_size * w / float(h))

    f_img = cv2.resize(src, (w, h), interpolation=interpolation)
    return f_img
