from core import IMAGE_HEIGHT, IMAGE_WIDTH


def write(img):
    pixels = img.reshape((IMAGE_WIDTH*IMAGE_HEIGHT, 3))
    print(f'P3')
    print(f'{IMAGE_WIDTH} {IMAGE_HEIGHT}')
    print(f'255')

    for r, g, b in pixels:
        print(f'{r} {g} {b}')
