import numpy


def genrate_perfect_note_wave(piano_note, duration, sr=22050, noise=0.001):
    f = lambda n: 2 ** ((n - 49) / 12.) * 440.
    F = f(piano_note)

    t = numpy.linspace(0, duration, duration * sr)
    y = numpy.sin(2 * numpy.pi * F * t)
    y += numpy.randn(len(y)) * noise

    return y, sr
