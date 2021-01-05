from .common import IMAGE_HEIGHT, IMAGE_WIDTH


def flatten(img):
    pl = img.reshape((IMAGE_WIDTH*IMAGE_HEIGHT, 3))
    return pl


def write(pl):
    print(f'P3')
    print(f'{IMAGE_WIDTH} {IMAGE_HEIGHT}')
    print(f'255')

    for r, g, b in pl:
        print(f'{r} {g} {b}')
