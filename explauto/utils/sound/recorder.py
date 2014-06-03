import time
from threading import Thread, Lock

import numpy as np
import pyaudio


class Recorder(Thread):

    chunk = 1024
    format_ = pyaudio.paInt16
    channels = 1
    #rate = 48000
    record_seconds = 5

    def __init__(self, device_index=0, verbose=False):
        super(Recorder, self).__init__()
        self.device_index = device_index
        self.recording = False
        self.running = True
        self.data = []
        self._data_lock = Lock()
        self._init_audio()
        self.start()
        self._nerrors = 0  # Count IOErrors

    def _init_audio(self):
        self._audio = pyaudio.PyAudio()
        self.rate = int(self._audio.get_device_info_by_index(self.device_index)[
                'defaultSampleRate'])
        self._stream = self._audio.open(
                format=self.format_,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index = self.device_index,
                output=True,
                frames_per_buffer=self.chunk)

    def run(self):
        while self.running:
            try:
                new_data = self._stream.read(self.chunk)
            except IOError:  # Have a better idea?
                self._nerrors += 1
                new_data = '\x00' * self.chunk
            if self.recording:
                # Append data
                self._data_lock.acquire()
                self.data.append(new_data)
                self._data_lock.release()

    def get_data(self):
        """Returns a copy of the recorded data.
        """
        self._data_lock.acquire()
        data = self.data
        self._data_lock.release()
        self.data = []
        return np.fromstring(''.join(data), 'int16')

    def exit(self):
        self.running = False
        self._stream.close()
        time.sleep(.01)
        self._audio.terminate()
        self.join()
