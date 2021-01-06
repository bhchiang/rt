from . import vec, Ray

ASPECT_RATIO = 16 / 9

VIEWPORT_HEIGHT = 2.0
VIEWPORT_WIDTH = ASPECT_RATIO * VIEWPORT_HEIGHT
FOCAL_LENGTH = 1

# camera
origin = vec.create()
horizontal = vec.create(VIEWPORT_WIDTH, 0, 0)
vertical = vec.create(0, VIEWPORT_HEIGHT, 0)
lower_left_corner = origin - (horizontal/2) - \
    (vertical/2) - vec.create(0, 0, FOCAL_LENGTH)


def shoot(u, v):
    begin = origin
    end = lower_left_corner + u*horizontal + v*vertical
    return Ray(origin=origin, direction=end - begin)
