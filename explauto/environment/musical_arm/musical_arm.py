from numpy import array, linspace, nonzero, hstack
from numpy import random

from .. import Environment
from ...utils import bounds_min_max
from ..simple_arm import SimpleArmEnvironment


# keyborad bounds


class MusicalArm(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length_ratio, noise, pitches, pitch_noises, box):
        Environment.__init__(self, m_mins, m_maxs,
                             hstack((s_mins, pitches[0] - pitch_noises[0])),
                             hstack((s_maxs, pitches[-1] + pitch_noises[-1])))
        self.arm = SimpleArmEnvironment(m_mins, m_maxs, s_mins, s_maxs, length_ratio, noise)
        self.pitches = pitches
        self.noises = pitch_noises
        self.x_min, self.y_min, self.x_max, self.y_max = box
        self.key_lims = linspace(self.y_min, self.y_max, len(self.pitches))

    def compute_motor_command(self, joint_pos):
        return bounds_min_max(joint_pos, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, joint_pos):
        x, y = self.arm.compute_sensori_effect(joint_pos)
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            note = 0  # no note
        else:
            note = nonzero(y < self.key_lims)[0][0]
        pitch = self.pitches[note]
        pitch += self.noises[note] * random.randn()
        return array([x, y, pitch])

    def plot_keyboard(self, ax, pitches=True, **kwargs_plot):
        plot_specs = {'color': 'black'}
        plot_specs.update(kwargs_plot)
        ax.plot([self.x_min, self.x_min, self.x_max, self.x_max, self.x_min],
                [self.y_min, self.y_max, self.y_max, self.y_min, self.y_min], **plot_specs)

        for i, y in enumerate(self.key_lims[:-1]):
            ax.plot([self.x_min, self.x_max], [y, y], **plot_specs)
            if pitches:
                ax.text(self.x_min + 0.1, y + 0.1, str(self.pitches[i + 1])) 
