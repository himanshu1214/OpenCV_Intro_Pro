import cv2


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


def resize_capture_image(capture, preferred_size):
    """
    this method is used to set the capture video and set the preferred size
    use the original capture in case of video failure
    :param capture:
    :param preferred_size:
    :return:
    """

    # set the dimensions
    w, h = preferred_size
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # discard for fluctuation
    capture.read()
    capture.read()

    # return the actual dimensions
    success, image = capture.read()
    if success and image is not None:
        h, w = capture.read()
    return (w, h)
