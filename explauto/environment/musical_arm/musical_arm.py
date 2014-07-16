from numpy import array, linspace, nonzero, hstack
from itertools import product, combinations

from numpy import random

from .. import Environment
from ...utils import bounds_min_max
from ..simple_arm import SimpleArmEnvironment


def is_in_box(box, pos):
    res = True
    for d, p in enumerate(pos):
        res &= p >= box[d][0] and p < box[d][1]
    return res


class MusicalArm(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length_ratio, noise, pitches, pitch_noises, boxes):
        Environment.__init__(self, m_mins, m_maxs,
                             hstack((s_mins, pitches[0] - pitch_noises[0])),
                             hstack((s_maxs, pitches[-1] + pitch_noises[-1])))
        self.arm = SimpleArmEnvironment(m_mins, m_maxs, s_mins, s_maxs, length_ratio, noise)
        self.pitches = pitches
        self.noises = pitch_noises
        # self.x_min, self.y_min, self.x_max, self.y_max = box
        # self.key_lims = linspace(self.y_min, self.y_max, len(self.pitches))
        self.boxes = boxes

    def compute_motor_command(self, joint_pos):
        return bounds_min_max(joint_pos, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, joint_pos):
        hand_pos = x, y = self.arm.compute_sensori_effect(joint_pos)
        to_play = [i for i, b in enumerate(self.boxes) if is_in_box(b, hand_pos)]
        note = 0 if not to_play else to_play[0]
        # if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
        #     note = 0  # no note
        # else:
        #     note = nonzero(y < self.key_lims)[0][0]
        pitch = self.pitches[note]
        pitch += self.noises[note] * random.randn()
        return array([x, y, pitch])

    def _plot_box(self, ax, i, **kwargs_plot):
        box = array(self.boxes[i]).T
        [ax.plot(*zip(s1, s2), color="b")
         for s1, s2 in list(combinations(product(*zip(*box)), 2))
         if (array(s1) == array(s2)).sum() == len(s1) - 1]


    def plot_boxes(self, ax, **kwargs_plot):
        for i in range(len(self.boxes)):
                self._plot_box(ax, i)
                coord = (array(self.boxes[i]).mean(axis=1))
                # ax.text(*coord, s=self.labels[i])

    # def plot_keyboard(self, ax, pitches=True, **kwargs_plot):
    #     plot_specs = {'color': 'black'}
    #     plot_specs.update(kwargs_plot)
    #     ax.plot([self.x_min, self.x_min, self.x_max, self.x_max, self.x_min],
    #             [self.y_min, self.y_max, self.y_max, self.y_min, self.y_min], **plot_specs)
    #
    #     for i, y in enumerate(self.key_lims[:-1]):
    #         ax.plot([self.x_min, self.x_max], [y, y], **plot_specs)
    #         if pitches:
    #             ax.text(self.x_min + 0.1, y + 0.1, str(self.pitches[i + 1]))
