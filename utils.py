from common import IMAGE_HEIGHT, IMAGE_WIDTH


def create_pixel_list(img):
    rshp = img.reshape((IMAGE_WIDTH*IMAGE_HEIGHT, 3))
    return rshp


def write_pixel_list(pxls):
    print(f'P3')
    print(f'{IMAGE_WIDTH} {IMAGE_HEIGHT}')
    print(f'255')

    for r, g, b in pxls:
        print(f'{r} {g} {b}')
