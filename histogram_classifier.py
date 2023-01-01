#!/usr/bin/ env python
import os

import cv2
import numpy
import scipy.io
import scipy.sparse


class HistogramClassifier:
    def __init__(self):
        self.verbose = False
        self.min_similarity_for_positive_label = 0.075
        self._channels = range(3)
        self._histsize = [256] * 3  # each color has 8 bit i.e. 256 values
        self._ranges = [0, 255] * 3
        self._references = {}  # maps the strings as keys to referenced histograms

    # convert into histogram using opncv and optionally convert into sparse matrix
    def _create_normalized_hist(self, image, sparse):

        # Create histogram
        hist = cv2.calcHist([image], self._channels, None, self._histsize, self._ranges)

        # Normalize histogram
        hist[:] = hist * (1.0 / numpy.sum(hist))

        # Convert to one Dimension for efficient storage
        hist = hist.reshape(16777216, 1)

        if sparse:
            hist = scipy.sparse.csc_matrix(hist)

        return hist

    # method to add the label "description" to the image (in sparse format) in push into list
    def add_reference(self, image, label):
        _hist = self._create_normalized_hist(image, True)

        if label not in self._references:
            self._references[label] = [_hist]
        else:
            self._references[label] += [_hist]

    # for the purpose of app ,  the image comes from filesystem, this method reads the image from the file-system
    # and is read in color scale and added into references list with label
    def add_reference_from_file(self, path, label):
        # Public method
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        self.add_reference(image, label)

    def classify(self, query_image, query_image_name=None):
        """
        this method computes the similarity for the query histogram versus the avg. references histogram and compares
        if all the similarity images are below the threshold than it will return 'Unknown'
        """
        query_hist = self._create_normalized_hist(query_image, False)
        b_label = "Unknown"
        b_similarity = self.min_similarity_for_positive_label
        if self.verbose:
            print("####          Here we begin classification       ##########")
            if query_image_name:
                print(f"Query image name : {query_image_name}")
        for label, hist_list in self._references.items():
            similarity = 0.0
            for hist in hist_list:
                similarity += cv2.compareHist(
                    hist.todense(), query_hist, cv2.HISTCMP_INTERSECT
                )
            similarity /= len(hist_list)
            if self.verbose:
                print(f"Similarity : {similarity} and label : {label}")
            if similarity > b_similarity:
                b_label = label
                b_similarity = similarity
            print(similarity, label)
        if self.verbose:
            print(
                "##########                  Classification ended here                ###############"
            )
        return b_label

    def classify_from_file(self, image_path, image_label=None):
        """
        this public method is used to get the image from filesystem and classify the file
        :return:
            classify the image based on the similarity thresh-hold
        """
        if not image_label:
            image_label = image_path
        hist_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return self.classify(hist_image, image_label)

    def serialize(self, path, compressed=False):
        """
        This method is used to read the reference hist images to/from disk by serializing
        :param path:
        :param compressed:
        :return: None
        """

        file = open(path, "wb")
        scipy.io.savemat(file, self._references, do_compression=compressed)

    def deserialize(self, path):
        """
        This method deserializes the histograms and removes metadata while loading
        :param path: serialized data path
        :return: None
        """
        file = open(path, "rb")
        self._references = scipy.io.loadmat(file)

        for key in list(self._references.keys()):
            value = self._references[key]
            if not isinstance(value, numpy.ndarray):  # deleting the metadata
                del self._references[key]
                continue
            self._references[key] = value[0]


def main():
    classifier = HistogramClassifier()
    classifier.verbose = True
    path = r"C:\Users\himan\OneDrive\Desktop\OpenCV-4-for-Secret-Agents-Second-Edition\Chapter002\images"
    list_files = os.listdir(path)
    for file in list_files:
        image_name = "".join(file.split(".")[:-1])
        classifier.add_reference_from_file(os.path.join(path, file), image_name)
        classifier.serialize("classifier.mat")
        classifier.deserialize("classifier.mat")

        print(
            classifier.classify_from_file(os.path.join(path, "dubai_damac_heights.jpg"))
        )
        print(
            classifier.classify_from_file(
                os.path.join(path, "communal_apartments_01.jpg")
            )
        )


if __name__ == "__main__":
    main()
