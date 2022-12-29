import numpy
import cv2
import wx

wx_major_version = int(wx.__version__.split('.')[0])

print(f"major_version: {wx_major_version}")

# CV2 - sets the colors of the image in BGR
# wxpython reads in RGB order
# Convert the color from CV2 to wxpython


def convert_color_fromcv2_towx(image):
    image_colr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image_colr.shape[:2]

    # read into bitmap
    if wx_major_version < 4:
        bitmap = wx.BitmapFromBuffer(w,h, image_colr)

    else:
        bitmap = wx.Bitmap.FromBuffer(w, h, image_colr)

    return bitmap
