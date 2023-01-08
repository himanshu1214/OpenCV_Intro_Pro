import os
import sys

# get the relative path to the app directory either its bundled as an app or not


def resource_path_resolver(relative_path):
    basepath = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(basepath, relative_path)
