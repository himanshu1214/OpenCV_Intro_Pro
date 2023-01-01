#! /usr/bin/env python

from py_ms_cognitive import PyMsCognitiveImageSearch

PyMsCognitiveImageSearch.SEARCH_IMAGE_BASE = (
    "https://api.bing.microsoft.com/v7.0/images/search"
)
import os
import pprint
import sys

import cv2

import request_utils


class ImageSearchSession:
    def __init__(self):
        self.verbose = False
        self._query = ""
        self._results = []
        self._offset = 0
        self._numResultsRequested = 0
        self._numResultsReceived = 0
        self._numResultsAvailable = 0

    # Adding getter methods

    @property
    def query(self):
        return self._query

    @property
    def numResultsRequested(self):
        return self._numResultsRequested

    @property
    def numResultsReceived(self):
        return self._numResultsReceived

    @property
    def numResultsAvailable(self):
        return self._numResultsAvailable

    def searchPrev(self):
        if self._offset == 0:
            return
        offset = self._offset + self._numResultsRequested
        self.search(self._query, offset)

    def searchNext(self):
        if self._offset + self._numResultsRequested >= self._numResultsAvailable:
            return

        offset = self._offset + self._numResultsRequested
        self.search(self._query, self._numResultsRequested, offset)

    def search(self, query, numResultsRequested=50, offset=0):
        bing_key = os.environ.get("BING_SEARCH_KEY")
        if not bing_key:
            sys.stderr.write("""undefined bing key""")
            return

        self._query = query
        self._numResultsRequested = numResultsRequested
        self._offset = offset
        params = {"color": "ColorOnly", "imageType": "Photo"}
        searchService = PyMsCognitiveImageSearch(bing_key, query, custom_params=params)
        searchService.current_offset = offset

        try:
            self._results = searchService.search(numResultsRequested, "json")
        except Exception as e:
            sys.stderr.write(f"Error as here: {e}")

            self._offset = 0
            self._numResultsReceived = 0
            return

        __json = searchService.most_recent_json

        self._numResultsReceived = len(self._results)
        if self._numResultsRequested < self._numResultsReceived:
            self._numResultsRequested = self._numResultsReceived
        self._numResultsAvailable = self._numResultsReceived

        self._numResultsAvailable = int(__json["totalEstimatedMatches"])

        if self.verbose:
            print("Received results of Bing image search for " '"%s":' % query)
            pprint.pprint(__json)

    def get_cv_image_and_url(self, index, useThumbnail=False):
        """
        extract the url from the bing api response and get the read the image as array
        :param index: the current index of the result
        :param useThumbnail:
        :return: image array
        """
        if index >= self._numResultsReceived:
            return None, None
        result = self._results[index]
        if useThumbnail:
            url = result.thumbnail_url
        else:
            url = result.content_url
        return request_utils.getcvImageFromUrl(url), url


def main():
    session = ImageSearchSession()
    session.verbose = True
    session.search("luxury condo sales")
    image, url = session.get_cv_image_and_url(0)
    cv2.imwrite("image.png", image)


if __name__ == "__main__":
    main()
