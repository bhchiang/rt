
ASPECT_RATIO = 16 / 9
IMAGE_WIDTH = 400
IMAGE_HEIGHT = int(IMAGE_WIDTH / ASPECT_RATIO)
SAMPLES_PER_PIXEL = 50
MAX_DEPTH = 50

# avoid circular dependencies
from . import vec
from .ray import Ray
from .camera import Camera




