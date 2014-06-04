import time
from numpy import hstack
from numpy import random

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
                 boxes, sound_samples, sound_analyser, analyser_noise, sound_mins, sound_maxs):

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

