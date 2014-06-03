import threading

import pyaudio
import numpy


class Recorder(object):
    def __init__(self, chunk=1024, rate=44100):
        self.running = threading.Event()
        self._data = ''

        self.chunk = chunk
        self.format = pyaudio.paInt16
        self._samplerate = rate
        self.channels = 1

        self.p = pyaudio.PyAudio()

    def start(self):
        if self.running.is_set():
            self.stop()

        self._data = ''
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.samplerate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

        self.running.set()
        self.t = threading.Thread(target=self._record)
        self.t.start()

    def stop(self):
        self.running.clear()

        if hasattr(self, 't'):
            self.t.join()

            self.stream.stop_stream()
            self.stream.close()

    @property
    def data(self):
        samplewidth = 8 * self.p.get_sample_size(self.format)

        data = numpy.fromstring(self._data, dtype=numpy.int16)
        normalized_data = data.astype(numpy.float64) / (2 ** samplewidth)
        return normalized_data

    @property
    def samplerate(self):
        return self._samplerate

    def _record(self):
        while self.running.is_set():
            self._data += self.stream.read(self.chunk)
