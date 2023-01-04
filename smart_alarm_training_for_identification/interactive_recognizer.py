import os
import sys
import threading

import cv2
import numpy
import wx
import wx_utils

import binascii_utils
import resize_utils


class InteractiveRecognizer(wx.Frame):
    """
    This class uses Frame subclass and houses the app main thread which updates for each GUI for each user action
    calculation for each video capture and identification and updating happen in the daemon thread

    Mainly using detection models and recognition models
    """

    def __init__(
        self,
        recognizer_path,
        cascade_path,
        scale_factor=1.3,
        min_neighbor=4,
        min_size_proportion=(0.25, 0.25),
        rect_color=(0, 255, 0),
        camera_device_id=0,
        image_size=(1280, 720),
        title="interactive recognizer",
    ):
        """

        :param recognizer_path: file contain recognizer model
        :param cascade_path: file with detection model
        :param scale_factor:used for searching faces at various scales, higher scale means faster search but lesser detection
        :param min_neighbor:neighbour that are needed to pass as faces, higher the val fewer detection and lower false positives
        :param min_size_proportion:ratio of camera resolution (640 x 480) to face size (120 x 120) with (0.25,0.25) to pass detection
        higher-val means lower detection
        :param rect_color: color of rectangle specified in BGR (opencv format)
        :param camera_device_id: device ID
        :param image_size: preffered image resolution
        :param title: app name
        """

        self.mirrored = True  # defaulted to true as camera feeds of image as intuitive
        self._running = True  # to track the app is running or closing, helpful for cleaning background thread
        self._capture = cv2.VideoCapture(camera_device_id)

        # resize to preferred dim or capture actual dim
        size = resize_utils.resize_capture_image(self._capture, image_size)
        self._image_width, self._image_height = size

        # capture and processing in two separate threads using thread locking (mutex)
        self._image = None
        self_gray_image = None
        self._equalized_gray_image = None

        self._image_from_buffer = None
        self._image_front_buffer_lock = threading.Lock()

        # detection and recognizer models related variables
        self._curr_detected_obj = None

        # for older recognizer model file path
        self._recognizer_path = recognizer_path

        # invoke recognizer model class
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()

        # read the model file (if exists) into model class
        if os.path.isfile(self._recognizer_path):
            self._recognizer.read(self._recognizer_path)
            self._recognizerTrained = True
        else:
            self._recognizerTrained = False

        self._detector = cv2.CascadeClassifier(cascade_path)
        self._scaleFactor = scale_factor
        self._minNeighbors = min_neighbor
        min_image_size = min(self._image_width, self._image_height)
        self._minSize = (
            int(min_image_size * min_size_proportion[0]),
            int(min_image_size * min_size_proportion[1]),
        )
        self._rectColor = rect_color

        # setting the GUI widgets (video panel, buttons, label, text field) and set their callbacks
        self._videoPanel = wx.Panel(self, size=size)
        self._videoPanel.Bind(wx.EVT_ERASE_BACKGROUND, self._onVideoPanelEraseBackground)


        # Setting the style, background colour, size and title
        style = (
            wx.CLOSE_BOX
            | wx.MINIMIZE_BOX
            | wx.CAPTION
            | wx.SYSTEM_MENU
            | wx.CAPTION
            | wx.CLIP_CHILDREN
        )
        super().__init__(self, None, title=title, style=style, size=size)
        self.SetBackgroundColour(wx.Colour(232, 232, 232))
        self.Bind(wx.EVT_CLOSE, self._onCloseWindow)

        # starting a background thread which captures the video and processs, detects and recognize
        # handling the compute intensive work in background for unblocking the GUI events
        self._captureThread = threading.Thread(target=self.run_capture_loop)
        self._captureThread.start()

    def _onCloseWindow(self, event):
        """
        when the window is close, stop the background thread.
        For trained recognition model, save it before destroying
        :param event:
        :return:
        """
        self._running = False
        self._captureThread.join()
        if self._recognizerTrained:
            model_dir = os.path.dirname(self._recognizer_path)
            if not os.path.isdir(self._recognizer_path):
                os.makedirs(model_dir)
            self._recognizer.write(self._recognizer_path)
        self.Destroy()

    def run_capture_loop(self):
        """
        this method is used to run async loop in the background which capture the image and detect & recognize
        the image and present to the GUI.
        then swapping of old buffer and new buffer image happens by acquiring mutex
        :return:
        """
        while self._running:
            success, self._image = self._capture.read(self._image)
            if self._image is not None:
                self._detect_and_recognize()
                if self.mirrored:
                    # flip the image i.e. mirror the image
                    self._image[:] = numpy.fliplr(self._image)

                    # swapping the image captured to front buffer and front to back buffer
                    self._image_front_buffer_lock.acquire()
                    self._image, self._image_from_buffer = self._image_from_buffer, self._image

                    # release the lock
                    self._image_front_buffer_lock.release()

                    # the image is drawn from the video into the front buffer
                    # send a refresh event to the video panel
                    self._videoPanel.Refresh()

    def _detect_and_recognize(self):
        """
        helper method which runs in the background thread and helps in detecting face
        using grayscale to create uniformity of image by removing the colored into gray scale
        :return:
        """
        self._gray_image = cv2.cvtColor(
            self._image, cv2.COLOR_BGR2GRAY, self._gray_image
        )
        self._equalized_gray_image = cv2.equalizeHist(
            self._gray_image, self._equalized_gray_image
        )

        # using Multiscale method to detect face and use green rectangle as boundary
        # return a list of rectangles which shows the bound of face
        detct = self._detector.detectMultiScale(self._equalized_gray_image,
                                                scaleFactor=self._scaleFactor,
                                                min_neighbor=self._minNeighbors,
                                                minSize=self._minSize)

        for x,y, w, h in detct:
            cv2.rectangle(self._image, (x, y), (x+w, y+h), self._rectColor, 1)

        if len(detct) >0:
            x, y, w,h = detct[0]
            # if atleast one face is detected, store detected face in equalized gray scale
            self._curr_detected_obj = cv2.equalizeHist(self._gray_image[y:y+h, x:x+w])
