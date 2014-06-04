import pygame
import time
import os


class SoundPlayer(object):
    def __init__(self, samplerate=44100):
        pygame.mixer.init(frequency=samplerate)

    def play(self, filename, wait=False):
        if not os.path.exists(filename):
            raise IOError("No such file or directory: '{}'".format(filename))

        s = pygame.mixer.Sound(filename)
        s.set_volume(1.0)
        length = s.get_length()

        s.play()

        if wait:
            time.sleep(length)

        return length


class Sampler(object):
    def __init__(self, sound_files, samplerate=44100):
        self.player = SoundPlayer(samplerate)
        self.sounds = sound_files

    def play(self, i, wait=False):
        return self.player.play(self.sounds[i], wait)

    def multiple_plays(self, sound_ids, wait=False):
        lengths = [self.play(i) for i in sound_ids]

        if wait:
            time.sleep(max(lengths))
