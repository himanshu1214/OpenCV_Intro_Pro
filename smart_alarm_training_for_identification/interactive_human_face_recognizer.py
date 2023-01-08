#!usr/bin/env python
import wx

import pyinstaller_utils
from interactive_recognizer import InteractiveRecognizer


def main():
    app = wx.App()
    recognizer_path = pyinstaller_utils.resource_path_resolver(
        "recognizers/lbph_human_faces.xml"
    )
    cascade_path = pyinstaller_utils.resource_path_resolver(
        "cascades/haarcascades_frontface.xml"
    )
    # cascade_path = pyinstaller_utils.resource_path_resolver('cascades/lbpcascades_frontalface.xml')

    interactive_recognizer = InteractiveRecognizer(
        recognizer_path, cascade_path, title="human recognizer app"
    )
    interactive_recognizer.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
