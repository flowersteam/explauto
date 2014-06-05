from numpy import hstack, random, array

import librosa

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
                 boxes,
                 sound_samples, sound_analyser, analyser_noise, sound_mins, sound_maxs,
                 internal_play_and_record):

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

        self._internal = internal_play_and_record

    def compute_motor_command(self, m_ag):
        return self.env.compute_motor_command(m_ag)

    def compute_sensori_effect(self, m_env):
        hand_pos = self.env.compute_sensori_effect(m_env)

        s, r = self.play_and_record(hand_pos)

        sound_value = self.analyser(s, r) + self.noise * random.randn()

        return hstack((hand_pos, sound_value))

    def play_and_record(self, hand_pos):
        to_play = [i for i, b in enumerate(self.boxes) if is_in_box(b, hand_pos)]

        if self.internal_play_and_record:
            sr = 22050
            y = [librosa.load(self.sampler.sounds[i], sr=sr)[0] for i in to_play]
            ml = min([len(yy) for yy in y])
            y = array([yy[:ml] for yy in y])
            r = sr
            s = y.mean(axis=0)

        else:
            self.recorder.start()
            self.sampler.multiple_plays(to_play, wait=True)
            self.recorder.stop()
            r = self.recorder.samplerate
            s = self.recorder.data

        return s, r

    @property
    def internal_play_and_record(self):
        return self._internal

    @internal_play_and_record.setter
    def internal_play_and_record(self, b):
        self._internal = b
