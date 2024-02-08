import tensorflow as tf
import numpy as np
import argparse
import sounddevice as sd
from time import time
from scipy.io.wavfile import write

class Spectrogram():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s):
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(
            audio, 
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        self.spectrogram = tf.abs(stft)

        return self.spectrogram

    def get_spectrogram_and_label(self, audio, label):
        
        audio = self.get_spectrogram(audio)

        return self.spectrogram, label

class MelSpectrogram():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency
    ):
        self.spectrogram_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        num_spectrogram_bins = self.spectrogram_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sampling_rate,
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency
        )

    def get_mel_spec(self, audio):
        spectrogram = self.spectrogram_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        return log_mel_spectrogram

    def get_mel_spec_and_label(self, audio, label):
        log_mel_spectrogram = self.get_mel_spec(audio)

        return log_mel_spectrogram, label

class VAD():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        dbFSthres,
        duration_thres
    ):
        self.frame_length_in_s = frame_length_in_s
        self.mel_spec_processor = MelSpectrogram(
            sampling_rate, frame_length_in_s, frame_length_in_s, num_mel_bins, lower_frequency, upper_frequency
        )
        self.dbFSthres = dbFSthres
        self.duration_thres = duration_thres

    def is_silence(self, audio):
        log_mel_spec = self.mel_spec_processor.get_mel_spec(audio)
        dbFS = 20 * log_mel_spec
        energy = tf.math.reduce_mean(dbFS, axis=1)

        non_silence = energy > self.dbFSthres
        non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
        non_silence_duration = (non_silence_frames + 1) * self.frame_length_in_s

        if non_silence_duration > self.duration_thres:
            return 0
        else:
            return 1
        

# just the device argument        
parser = argparse.ArgumentParser()
parser.add_argument('--device', type = int, default = 1)
args = parser.parse_args()

# specifications of the exercise 
resolution = 'int16'
sample_rate = 16000
channels = 1
"""
 audio_buffer will store 1 sec of audio
 so at 16kHz we have a tensor of length 16000 
"""
global audio_buffer 
audio_buffer = tf.zeros(shape=(16000,), dtype=np.float32) 

# creating the vad processor with the parameter extracted from ex 1.1 
vad_processor = VAD(sampling_rate = 16000,
                    frame_length_in_s = 0.032,
                    num_mel_bins = 12,
                    lower_frequency = 0,
                    upper_frequency = 8000,
                    dbFSthres = -42,
                    duration_thres = 0.05
                    )

"""
this function create a tensor from  a numpy array and 
normalize it 

"""
def get_tf(indata):
       
    tf_indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    audio_tensor = tf.squeeze(tf_indata)
    audio_float32 = tf.cast(audio_tensor, tf.float32)
    audio_normalized = audio_float32 / tf.int16.max   # normalization 
    return audio_normalized



blocksize = 8000 
"""
 this function is called every blocksize
 in our case we have a sampling rate of 16kHz and we need to check 
 every 0.5 second the last second of recorded audio, so the blocksize is set to 8k
"""
def callback(indata, frames, callback_time, status):
    global audio_buffer 

    tf_indata = get_tf(indata)
    """
    |  this update audio_buffer with the new indata 
    V  like: ['old 0.5sec', 'new0.5sec']
    """
    audio_buffer = tf.concat([audio_buffer[8000:], tf_indata], axis=0) 

    if vad_processor.is_silence(audio_buffer):
        print('Silence')
    else:
        print('Voice') 
        timestamp = time()
        # saving the audio file
        write(f'./{timestamp}.wav', sample_rate, audio_buffer.numpy()) 



#  loop function for the recording 
print("Press Q to Stop the recording and exit!")

with sd.InputStream(device = args.device, channels = channels, dtype = resolution, 
                    samplerate = sample_rate, blocksize = blocksize, callback = callback):
    while True:
        key = input()
        if key in ('q', 'Q'):
            print('Stop recording.')
            break
