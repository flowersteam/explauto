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

        s.play()

        if wait:
            time.sleep(s.get_length())


class Sampler(object):
    def __init__(self, sound_files, samplerate=44100):
        self.player = SoundPlayer(samplerate)
        self.sounds = sound_files

    def play(self, i, wait=False):
        self.player.play(self.sounds[i], wait)

    def multiple_plays(self, *sound_ids):
        for i in sound_ids:
            self.play(i)
