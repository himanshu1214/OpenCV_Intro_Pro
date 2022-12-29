import requests
import cv2
import sys
import numpy
from dotenv import load_dotenv

load_dotenv()

HEADERS = {'User-Agent': 'Mozilla/5.0' \
                          '(Macintosh; Intel Mac OS X 10.9; rv:25.0) ' \
                          'Gecko/20100101 Firefox/25.0'
           }



def validate_response(response):
    """"""
    status = response.status_code
    if status == 200:
        return True

    sys.stderr.write(f"Error code received: {status} for URL :  {response.url}")
    return False


def getcvImageFromUrl(url):
    """"""
    response = requests.get(url, headers=HEADERS)

    if not validate_response(response):
        return None
    image_data = numpy.frombuffer(response.content, numpy.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        sys.stderr.write('Failed')

    print("got the correct image")
    return image


def main():
    image = getcvImageFromUrl('http://nummist.com/images/ceiling.gaze.jpg')
    if image is not None:
        cv2.imwrite('image.png', image)

if __name__ == '__main__':
    main()