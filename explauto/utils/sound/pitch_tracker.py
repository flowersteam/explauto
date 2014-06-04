#!/usr/bin/env python
'''
CREATED:2013-12-09 00:02:54 by Brian McFee <brm2132@columbia.edu>

Estimate the tuning (deviation from A440) of a recording.

Usage: ./tuning.py [-h] input_file
'''

import numpy as np

import librosa


def estimate_tuning(y, sr):
    # Get the instantaneous-frequency pitch track
    pitches, magnitudes, D = librosa.feature.ifptrack(y, sr)

    # Just track the pitches associated with high magnitude
    pitches = pitches[magnitudes > np.median(magnitudes)]

    tuning = librosa.feature.estimate_tuning(pitches)
    # print '%+0.2f cents' % (100 * tuning)

    return tuning


def get_key(y, sr):
    C = librosa.feature.chromagram(y=y, sr=sr)
    return C.argmax(axis=0).mean() / 12.
