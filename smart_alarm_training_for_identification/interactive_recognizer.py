import os
import sys
import threading

import cv2
import numpy
import wx

import binascii_utils
import resize_utils
import wx_utils


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
        self._gray_image = None
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
        self._videoPanel.Bind(
            wx.EVT_ERASE_BACKGROUND, self._on_video_panel_erase_background
        )

        # Setting the style, background colour, size and title
        style = (
            wx.CLOSE_BOX
            | wx.MINIMIZE_BOX
            | wx.CAPTION
            | wx.SYSTEM_MENU
            | wx.CAPTION
            | wx.CLIP_CHILDREN
        )
        wx.Frame.__init__(self, None, title=title, style=style, size=size)
        self.SetBackgroundColour(wx.Colour(232, 232, 232))
        self.Bind(wx.EVT_CLOSE, self._onCloseWindow)

        # add a callback for esc key, given key is not GUI widget -> no Bind method available for callback
        # Bind a new menu item and callback to the interactive recognizer instance
        # map a keyboard shortcut to the menu event using class wx.AcceleratorTable
        quit_command_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self._on_quit_command, id=quit_command_id)
        accelerator_table = wx.AcceleratorTable(
            [(wx.ACCEL_NORMAL, wx.WXK_ESCAPE, quit_command_id)]
        )
        self.SetAcceleratorTable(accelerator_table)

        # add a video panel , text field, panel and label
        self._videoPanel = wx.Panel(self, size=size)
        self._videoPanel.Bind(
            wx.EVT_ERASE_BACKGROUND, self._on_video_panel_erase_background
        )  # binding the callback to erase
        self._videoPanel.Bind(
            wx.EVT_PAINT, self._on_video_panel_paint
        )  # bind callback to set the images
        self._videoBitmap = None

        # add reference Txt control button
        self._referenceTextCtrl = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self._referenceTextCtrl.SetMaxLength(4)  # max length is 4 chars
        self._referenceTextCtrl.Bind(wx.EVT_KEY_UP, self._on_reference_text_ctrl_key_up)

        self._predictionStaticText = wx.StaticText(self)

        #  add newline char
        self._predictionStaticText.SetLabel("\n")

        # add update model button
        self._updateModelButton = wx.Button(self, label="Add to model")
        self._updateModelButton.Bind(wx.EVT_BUTTON, self._update_model)

        # add clear model button
        self._clearModelButton = wx.Button(self, label="Clear Model")
        self._clearModelButton.Bind(wx.EVT_BUTTON, self._clear_model)
        if (
            not self._recognizerTrained
        ):  # if model doesnot exist , disable the clear model buttton
            self._clearModelButton.Disable()

        # Putting up the controls
        border = 12
        controls_sizer = wx.BoxSizer(wx.HORIZONTAL)
        controls_sizer.Add(
            self._referenceTextCtrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border
        )
        controls_sizer.Add(
            self._updateModelButton, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border
        )
        controls_sizer.Add(self._predictionStaticText, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_sizer.Add((0, 0), 1)  # spacer
        controls_sizer.Add(self._clearModelButton, 0, wx.ALIGN_CENTER_VERTICAL)

        root_sizer = wx.BoxSizer(wx.VERTICAL)
        root_sizer.Add(self._videoPanel)
        root_sizer.Add(controls_sizer, 0, wx.EXPAND | wx.ALL, border)
        self._setSizerAndFit(root_sizer)

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

    def _on_quit_command(self, event):
        """
        callback method to close the window and attaches the Esc
        :param event:
        :return:
        """
        self.Close()

    def _on_video_panel_erase_background(self):
        """"""
        pass

    def _on_reference_text_ctrl_key_up(self):
        """
        this method is callback based on the text for image detected and chose to enable or disable add to model button
        :return:
        """
        self._enable_or_disable_update_model_button()

    def _update_model(self, event):
        """
        this method is used as a callback, provides training data to the recognition model. Either train model for no prior
        training data or update
        :return:
        """
        # get label from
        label_as_str = self._referenceTextCtrl.GetValue()
        label_as_int = binascii_utils.four_char_to_int(label_as_str)
        src = [self._curr_detected_obj]
        labels = numpy.array([label_as_int])

        # check if model exist using the trained model flag
        if self._recognizerTrained:
            self._recognizer.update(src, labels)
        else:
            self._recognizer.train(src, labels)
            self._recognizerTrained = True
            # enable clear model here
            self._clearModelButton.Disable()

    def _clear_model(self, event=None):
        """
        the callback method will delete the existing model and creates a new one and disable the delete button
        :return:
        """
        self._recognizerTrained = False
        self._clearModelButton.Disable()
        if os.path.isfile(self._recognizer_path):
            os.remove(self._recognizer_path)
        self._recognizer = (
            cv2.face.LBPHFaceRecognizer_create()
        )  # create the new untrained model

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
                    self._image, self._image_from_buffer = (
                        self._image_from_buffer,
                        self._image,
                    )

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
        detct = self._detector.detectMultiScale(
            self._equalized_gray_image,
            scaleFactor=self._scaleFactor,
            min_neighbor=self._minNeighbors,
            minSize=self._minSize,
        )

        for x, y, w, h in detct:
            cv2.rectangle(self._image, (x, y), (x + w, y + h), self._rectColor, 1)

        if len(detct) > 0:
            x, y, w, h = detct[0]
            # if atleast one face is detected, store detected face in equalized gray scale
            # equalized image is based on the cropped image for better avg local contrast instead of whole image
            self._curr_detected_obj = cv2.equalizeHist(
                self._gray_image[y : y + h, x : x + w]
            )

            # if model exist even for 1 image trained, then model will return 2 integer name and distance (confidence value)
            if self._recognizerTrained:
                try:
                    label_as_int, distance = self._recognizer.predict(
                        self._curr_detected_obj
                    )
                    label_as_str = binascii_utils.int_to_four_char(label_as_int)
                    self._show_message(
                        f"Looks similar to the image :{label_as_str} and distance is : {distance}"
                    )
                except cv2.error:
                    print >> sys.stderr, "recreating model due to err"
                    self._clear_model()

            else:
                self._show_instructions()
        else:
            self._curr_detected_obj = None  # set current object detected to None
            if self._recognizerTrained:  # if model exist then print message on screen
                self._clear_message()
            else:  # show instructions
                self._show_instructions()

        # adding the enable/disable add to model button
        self._enable_or_disable_update_model_button()

    def _enable_or_disable_update_model_button(self):
        """this method is implemented based on the image is detected, if detected and text box is not empty
        then show enable the button
        """
        label_as_str = self._referenceTextCtrl.GetValue()
        if len(label_as_str) < 1 or self._curr_detected_obj is None:
            self._updateModelButton.Disable()  # implemented all button in the app initiation
        else:
            self._updateModelButton.Enable()

    def _on_video_panel_erase_background(self, event):
        """not doing anything just passing the previous image , draw over the old video frame"""
        pass

    def _on_video_panel_paint(self, event):
        """
        In thread safe manner - use the front image buffer and convert it into bitmap and finally show it to GUI
        """
        self._image_front_buffer_lock.acquire()
        if self._image_from_buffer is None:
            self._image_front_buffer_lock.release()
            return
        # Convert the image into wxPython bitmap
        self._videoBitmap = wx_utils.convert_color_fromcv2_towx(self._image_from_buffer)

        self._image_front_buffer_lock.release()

        # Show the bitmap
        dc = wx.BufferedPaintDC(self._videoPanel)
        dc.DrawBitmap(self._videoBitmap, 0, 0)

    # helper methods to show message
    def _show_instructions(self):
        """"""
        self._show_message(
            "when object is highlighted, input the name max four chars \n"
            "and click add to model"
        )

    def _clear_message(self):
        """"""
        self._show_message("\n")

    def _show_message(self, message):
        """"""
        wx.CallAfter(self._predictionStaticText, message)
