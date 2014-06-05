# import time
from os.path import basename
from itertools import product, combinations
from numpy import hstack, random, array

from ...utils.sound.recorder import Recorder
from ...utils.sound.player import Sampler
from .. import Environment

record_time = 2  # in seconds


def is_in_box(box, pos):
    res = True
    for d, p in enumerate(pos):
        res &= p >= box[d][0] and p < box[d][1]
    return res


class MusicEnvironment(Environment):
    
    use_process = False

    def __init__(self, base_environment,
                 boxes, sound_samples, sound_analyser, analyser_noise, sound_mins, sound_maxs, sample_labels=None):

        Environment.__init__(self, base_environment.conf.m_mins,
                             base_environment.conf.m_maxs,
                             hstack((base_environment.conf.s_mins, sound_mins)),
                             hstack((base_environment.conf.s_maxs, sound_maxs)))

        self.env = base_environment
        self.boxes = boxes
        self.sampler = Sampler(sound_samples)
        self.recorder = Recorder()
        self.analyser = sound_analyser
        self.noise = analyser_noise
        if sample_labels is not None:
            self.labels = sample_labels
        else:
            self.labels = [basename(s) for s in sound_samples]

    def compute_motor_command(self, m_ag):
        return self.env.compute_motor_command(m_ag)

    def compute_sensori_effect(self, m_env):
        hand_pos = self.env.compute_sensori_effect(m_env)
        self.recorder.start()
        to_play = [i for i, b in enumerate(self.boxes) if is_in_box(b, hand_pos)]
        self.sampler.multiple_plays(to_play, wait=True)
        self.recorder.stop()
        r = self.recorder.samplerate
        s = self.recorder.data
        sound_value = self.analyser(s, r) + self.noise * random.randn()
        return hstack((hand_pos, sound_value))

    def _plot_box(self, ax, i, **kwargs_plot):
        box = array(self.boxes[i]).T
        [ax.plot(*zip(s1, s2), color="b")
         for s1, s2 in list(combinations(product(*zip(*box)), 2))
         if (array(s1) == array(s2)).sum() == len(s1) - 1]


    def plot_boxes(self, ax, **kwargs_plot):
        for i in range(len(self.boxes)):
                self._plot_box(ax, i)
                # ax.text(*(array(self.boxes[i])[:, 0]), text='test')

