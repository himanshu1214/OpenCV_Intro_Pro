import binascii


def four_char_to_int(s):
    """
    utility method to convert the char to 32 bit integer
    :return:
    """
    return int(binascii.hexlify(bytearray(s, "ascii")), 16)


def int_to_four_char(i):
    """
    utility method to convert the int to four char
    :return chars
    """
    return binascii.unhexlify(format(i, "x")).decode("ascii")
