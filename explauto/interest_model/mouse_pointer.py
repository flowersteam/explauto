from numpy import array

from .interest_model import InterestModel
from ..io.mouse_pointer import MousePointer as MP


class MousePointer(InterestModel):
    def __init__(self, conf, expl_dims, width, height):
        InterestModel.__init__(self, expl_dims)
        self.pointer = MP(width, height)

    def sample(self):
        return array(self.pointer.xy)

    def update(xy, ms):
        pass

interest_models = {'mouse_pointer_beta': (MousePointer,
                                         {'default': {'width': 320,
                                                      'height': 240}})}
