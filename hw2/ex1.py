import tensorflow as tf
import numpy as np
import argparse
import sounddevice as sd
import time
import psutil
import redis
import uuid




class RedisClient():
    def __init__(self,host,port,username,password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.redis_client = redis.Redis(host = self.host, port = self.port, username = self.username, password = self.password)
        self.mac_address = hex(uuid.getnode())

    def is_connected(self):
        is_connected = self.redis_client.ping()
        print(f'Redis Connected: {is_connected} \n')

    def create_key(self, key_name, retention_time_ms = None):
        try:
            self.redis_client.ts().create(key_name, retention_msecs=retention_time_ms)
        except redis.ResponseError:
            pass

    def add_ts_battery_data(self, stop = False):
        """
        parameters : duration in seconds

        This function will acquire data about percentage of battery level and power plugged for the specified duration and store those data on the redis timeseries.
        If no value is provided, the acquisition will continue without a fixed limit
        The function prints the execution time
        """
        if not stop:
            timestamp = time.time()
            timestamp_ms = int(timestamp * 1000)
            battery_level = psutil.sensors_battery().percent
            power_plugged = int(psutil.sensors_battery().power_plugged)
        
            self.redis_client.ts().add(f'{self.mac_address}:battery', timestamp_ms, battery_level)
            self.redis_client.ts().add(f'{self.mac_address}:power', timestamp_ms, power_plugged)


            
           


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
        


class MFCC():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        num_coefficients
    ):
        self.mel_processor = MelSpectrogram(sampling_rate,
                                            frame_length_in_s,
                                            frame_step_in_s,
                                            num_mel_bins,
                                            lower_frequency,
                                            upper_frequency
                                            )
        self.num_coeff = num_coefficients                                    
                                        

    def get_mfccs(self, audio):
        log_mel_spectrogram = self.mel_processor.get_mel_spec(audio)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :self.num_coeff]  
        return mfccs

    def get_mfccs_and_label(self, audio, label):
        mfccs = self.get_mfccs(audio)
        return mfccs, label



def get_tf(indata):
    """
    this function create a tensor from  a numpy array and 
    normalize it 
    """
    tf_indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    audio_tensor = tf.squeeze(tf_indata)
    audio_float32 = tf.cast(audio_tensor, tf.float32)
    audio_normalized = audio_float32 / tf.int16.max   # normalization 
    return audio_normalized



blocksize = 8000 

def callback(indata, frames, callback_time, status):
    """
    this function is called every blocksize
    in our case we have a sampling rate of 16kHz and we need to check 
    every 0.5 second the last second of recorded audio, so the blocksize is set to 8k
    """
    global audio_buffer,redis_client,stop
   
    tf_indata = get_tf(indata)
    """
    |  this update audio_buffer with the new indata 
    V  like: ['old 0.5sec', 'new0.5sec']
    """
    audio_buffer = tf.concat([audio_buffer[8000:], tf_indata], axis=0)
    redis_client.add_ts_battery_data(stop=stop)
    if vad_processor.is_silence(audio_buffer):
        print('Silence')
    else:
        print('Voice') 

        # Get and reshape the mfccs array
        audio_data = mfcc_processor.get_mfccs(audio_buffer)
        audio_data =  tf.reshape(audio_data, (1, 61, 8, 1))
        
        interpreter.set_tensor(input_details[0]['index'], audio_data)
        interpreter.invoke()
        #Get the output
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        
        # If the largest value between the probability of "yes" (output[0]) and the probability of "no" (output[1]) is <= 99%, remain in the current state
        if not max(output[0],output[1]) <= 0.99:
            if output[0] > 0.99:
                #Start monitoring
                print('yes')
                stop = False
                
            if output[1] > 0.99:
                #Stop monitoring
                print('no')
                stop = True 

        
        


if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=int)
    parser.add_argument('--user', type=str)
    parser.add_argument('--password', type=str)
    args = parser.parse_args()


    resolution = 'int16'
    sample_rate = 16000
    channels = 1

    

    #Initalize Reds client
    global redis_client
    redis_client = RedisClient(host=args.host, port=args.port, username=args.user, password=args.password)
    mac_address = redis_client.mac_address
    redis_client.is_connected()
    redis_client.create_key(f'{mac_address}:battery', retention_time_ms= 2 * 60 * 1000)
    redis_client.create_key(f'{mac_address}:power', retention_time_ms=2 * 60 * 1000)



    """
    audio_buffer will store 1 sec of audio
    so at 16kHz we have a tensor of length 16000 
    """
    global audio_buffer, stop
    audio_buffer = tf.zeros(shape=(16000,), dtype=np.float32) 

    
    model_path = './HW2_Team2/model2.tflite'
    stop = True
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    batch_size = input_details[0]['shape'][0]
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    print("Input Shape:", input_shape)


    PREPROCESSING_ARGS = {
    'sampling_rate': 16000,
    'frame_length_in_s': 0.032,
    'frame_step_in_s': 0.016,
    'num_mel_bins': 14,
    'lower_frequency': 0,
    'upper_frequency': 4000,
    'num_coefficients': 8,
}
    
    
    mfcc_processor = MFCC(**PREPROCESSING_ARGS)

    # creating the vad processor with the parameter extracted from ex 1.1 
    vad_processor = VAD(sampling_rate = 16000,
                        frame_length_in_s = 0.032,
                        num_mel_bins = 12,
                        lower_frequency = 0,
                        upper_frequency = 8000,
                        dbFSthres = -42,
                        duration_thres = 0.05
                        )




    #  Loop function for the recording 
    print("Press Q to Stop the recording and exit!")

    with sd.InputStream(device = args.device, channels=channels, dtype = resolution, 
                        samplerate = sample_rate, blocksize = blocksize, callback = callback):
        while True:
            key = input()
            if key in ('q', 'Q'):
                print('Stop recording.')
                break
