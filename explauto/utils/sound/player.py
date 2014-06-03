import pygame
import time


class SoundPlayer(object):
    def __init__(self, samplerate=44100):
        pygame.mixer.init(frequency=samplerate)

    def play(self, filename, wait=False):
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
