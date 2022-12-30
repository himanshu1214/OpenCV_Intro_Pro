#!/usr/bin/env python

import os

import numpy
import cv2

import threading
import wx

from histogram_classifier import HistogramClassifier
from image_search_session import ImageSearchSession
import pyinstaller_utils
import cvResizeAspectFill
import wx_utils


class Luxocator(wx.Frame):
    # Subclassing the wx.Frame class

    def __init__(self, classifier_path, max_image_size=768):
        style = wx.CLOSE_BOX | wx.MINIMIZE_BOX | wx.CAPTION | wx.SYSTEM_MENU | wx.CLIP_CHILDREN
        wx.Frame.__init__(self, None, title='Luxocator', style=style)
        self.SetBackgroundColour(wx.Colour(232, 232, 232))
        self._maxImageSize = max_image_size
        border = 12
        default_query_image = 'luxury condo sales'
        self._index = 0

        # Begin image search session object
        self._session = ImageSearchSession()
        self._session.verbose = False
        self._session.search(default_query_image)

        # image classifier object
        self._classifier = HistogramClassifier()
        self._classifier.verbose = True
        self._classifier.deserialize(classifier_path)

        self.Bind(wx.EVT_SIZE, self._onCloseWindow)

        # quit_command = wx.NewId()
        # self.Bind(wx.EVT_MENU, self._onQuitCommand, id=quit_command)
        # accelerator_table = wx.AcceleratorTable([(wx.ACCEL_NORMAL, wx.WXK_ESCAPE, quit_command)])
        # self.SetAcceleratorTable(accelerator_table)

        # Add button controls for text field, search button and cancel button

        self._searchCtrl = wx.SearchCtrl(self,
                                         size=(self._maxImageSize / 3, -1),
                                         style=wx.TE_PROCESS_ENTER)

        # Sets the new text control value.
        self._searchCtrl.SetValue(default_query_image)
        self._searchCtrl.Bind(wx.EVT_TEXT_ENTER,self._onSearchEntered) # binds to look the criteria
        self._searchCtrl.Bind(wx.EVT_SEARCHCTRL_SEARCH_BTN,self._onSearchEntered)
        self._searchCtrl.Bind(wx.EVT_SEARCHCTRL_CANCEL_BTN, self._onSearchCancelled)

        # set label
        self._labelStaticText = wx.StaticText(self)

        # set previous button
        self._prevButton = wx.Button(self, label='Prev')
        self._prevButton.Bind(wx.EVT_BUTTON, self._onPrevButtonClicked)

        # set next button
        self._nextButton = wx.Button(self, label='Next')
        self._nextButton.Bind(wx.EVT_BUTTON, self._onNextButtonClicked)

        # bitmap
        self._staticBitmap = wx.StaticBitmap(self)

        # Defining horizontal layout for search control on the left - label in middle - prev + next on right
        controls_sizer = wx.BoxSizer(wx.HORIZONTAL)
        controls_sizer.Add(self._searchCtrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border)
        controls_sizer.Add((0, 0), 1)  # spacer
        controls_sizer.Add(self._labelStaticText, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_sizer.Add((0, 0), 1)  # spacer
        controls_sizer.Add(self._prevButton, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, border)
        controls_sizer.Add(self._nextButton, 0, wx.ALIGN_CENTER_VERTICAL)

        # Defining bitmap layout using another wx.BoxSizer instance
        self._rootSizer = wx.BoxSizer(wx.VERTICAL)
        self._rootSizer.Add(self._staticBitmap,0, wx.Top | wx.LEFT | wx.RIGHT, border)
        self._rootSizer.Add(controls_sizer, 0, wx.EXPAND | wx.ALL, border)
        self.SetSizerAndFit(self._rootSizer)
        self._updateImageAndControls()

        # adding getter and setter property for verbose property of HistogramClassifier and ImageSearchSession class

    @property
    def verbose_search_session(self):
        return self._session.verbose

    @verbose_search_session.setter
    def verbose_search_session(self, value):
        self._session.verbose = value

    @property
    def verbose_histogram_classifier(self):
        return self._classifier.verbose

    @verbose_histogram_classifier.setter
    def verbose_histogram_classifier(self, value):
        self._classifier.verbose = value

    # defining callbacks
    def _onCloseWindow(self, event):
        """cleans up the application
        """
        self.Destroy()

    def _onQuitCommand(self, event):
        """
        connected Esc key on the callback,  closes the window
        :return:
        """
        self.Close()

    def _onSearchEntered(self, event):
        """
        searches the image asynchronously using image_search session
        :return:
        """
        query = event.GetString()
        if len(query) < 1:
            return

        self._session.search(query)
        self._index = 0
        self._updateImageAndControls()

    def _onSearchCancelled(self, event):
        """
        clears the seach text
        :param event:
        :return:
        """
        self._searchCtrl.Clear()

    def _onNextButtonClicked(self, event):
        """
        callback method based on the next image is available or not
        :return:
        """
        self._index += 1
        if self._index < self._session._offset + self._session.numResultsReceived - 1:
            self._session.searchNext()
        self._updateImageAndControls()

    def _onPrevButtonClicked(self):
        """
        callback method
        :return:
        """
        self._index -= 1
        if self._index < self._session._offset:
            self._session.searchNext()
        self._updateImageAndControls()

    def _disableControls(self):
        """
        disable search control, prev button , next button
        :return:
        """
        self._searchCtrl.Disable()
        self._prevButton.Disable()
        self._nextButton.Disable()

    def _enableControls(self):
        """
        enable search Controls , next button - (if not at the last image)
        , prev buttom - (if not a first image)
        :return:
        """
        self._searchCtrl.Enable()
        if self._index > 1:
            self._prevButton.Enable()
        if self._session.numResultsReceived - self._index > 1:
            self._nextButton.Enable()

    def _updateImageAndControls(self):
        """
        this method disable search controls and prev and next button during the image is loaded
        so not entertaining any new search query
        :return:
        """
        self._disableControls()
        # show busy cursor
        wx.BeginBusyCursor()

        # run image in background thread
        threading.Thread(target=self._updateImageAndControlsAsync).start()

    def _updateImageAndControlsAsync(self):
        """

        :return:
        """
        if self._session.numResultsReceived == 0:
            image = None
            label = 'No results found'
        else:
            image, url = self._session.get_cv_image_and_url(self._index % self._session.numResultsRequested)
            if image is None:
                label = 'No image found'
            else:
                # we received the image , now classify
                label = self._classifier.classify(image,  url)

                # resize the image using autofill to display in an appropriate size
                image = cvResizeAspectFill.resize_image(image, self._maxImageSize)

        # Update GUI in the main thread
        wx.CallAfter(self._updateImageAndControlsResync,  image, label)

    def _updateImageAndControlsResync(self, image, label):
        """
        synchronous method to remove the busy cursor and create wxPython bitmap format
        :arg
            image: opencv format image
            label:

        :return:
        """
        # hide the busy cursor
        wx.BusyCursor()
        if image is None:
            # return the black background
            bitmap = wx.Bitmap(self._maxImageSize, self._maxImageSize / 2)

        else:
            # convert the image into pybitmap format
            bitmap = wx_utils.convert_color_fromcv2_towx(image)

        # show the bitmap
        self._staticBitmap.SetBitmap(bitmap)

        # show label
        self._labelStaticText.SetLabel(label)

        # resize Sizer and Frame
        self._rootSizer.Fit(self)

        # Re-enable Controls
        self._enableControls()

        # Refresh
        self.Refresh()


def main():
    """

    :return:
    """
    # os.environ['REQUESTS_CA_BUNDLE'] = pyinstaller_utils.resource_path_resolver('cacert.pem')
    app = wx.App()
    luxocator = Luxocator(pyinstaller_utils.resource_path_resolver('classifier.mat'))
    luxocator.Show()
    app.MainLoop()

if __name__=='__main__':
    main()