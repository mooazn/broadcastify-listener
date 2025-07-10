import audioop
import collections
from datetime import datetime, timedelta
import os
import os.path
import queue
import subprocess
import sys
import threading
import time
import wave
import shutil
import whisper
import ffmpeg
import gflags
import sounddevice as sd
import numpy as np
import termcolor
import webrtcvad
import warnings
from text_summarizer import q

# credit here:
# https://gist.githubusercontent.com/wiseman/8547eee19421d69a838dc951e432071a/raw/0c4bf1a2a9adae0addce4dd76a8eea9abd10b087/broadcastify_listen.py

# my program uses a different voice recognition library and is quite modified

warnings.filterwarnings("ignore")

FLAGS = gflags.FLAGS

gflags.DEFINE_boolean(
    'show_ffmpeg_output',
    False,
    'Show ffmpeg\'s output (for debugging).')
gflags.DEFINE_string(
    'wav_output_dir',
    'wavs',
    'The directory in which to store wavs.')
gflags.DEFINE_string(
    'txt_output_dir',
    'txt',
    'The directory in which plain transcripts are stored.'
)

FLAGS(sys.argv)

def print_status(fmt, *args):
    sys.stdout.write('\r%s\r' % (75 * ' '))
    print(fmt % args)


def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        next(cr)
        return cr
    return start


@coroutine
def broadcast(targets):
    while True:
        item = (yield)
        for target in targets:
            target.send(item)

TS_FORMAT = '%Y%m%d-%H%M%S.%f'

def timestamp_str(dt):
    return dt.strftime(TS_FORMAT)[:-3]


class _BingRecognizer(object):
    def __init__(self, model_name='base'):
        self.model = whisper.load_model(model_name)

    def recognize_file(self, path):
        result = self.model.transcribe(path)
        return result["text"]

def start_daemon(callable, *args):
    t = threading.Thread(target=callable, args=args)
    t.daemon = True
    t.start()

def start_cleanup(cb):
    t = threading.Thread(target=cb)
    t.daemon = True
    t.start()

def remove_wav():
    time.sleep(60)
    while True:
        wav_files = os.listdir(FLAGS.wav_output_dir)
        for file_name in wav_files:
            parsed_time = datetime.strptime(file_name[:-4], TS_FORMAT)
            current_time = datetime.now()
            time_difference = current_time - parsed_time
            if time_difference >= timedelta(minutes=5):
                wav_file_path = os.path.join(FLAGS.wav_output_dir, file_name)
                os.remove(wav_file_path)
        time.sleep(300)

def queue_get_nowait(q):
    value = None
    try:
        value = q.get_nowait()
    except queue.Empty:
        pass
    return value


class _AudioFrame(object):
    def __init__(self, data=None, timestamp=None, sample_rate=None):
        self.data = data
        self.timestamp = timestamp
        self.sample_rate = sample_rate

    def __str__(self):
        return '<AudioFrame %s bytes (%s s)>' % (
            len(self.data),
            self.duration())

    def __repr__(self):
        return str(self)

    def duration(self):
        return len(self.data) / (2.0 * self.sample_rate)

    @staticmethod
    def coalesce(frames):
        "Coalesces multiple frames into one frame. Order is important."
        if not frames:
            return None
        frame = _AudioFrame(
            b''.join([f.data for f in frames]),
            sample_rate=frames[0].sample_rate,
            timestamp=frames[0].timestamp)
        return frame


@coroutine
def play_audio_co():
    stream = sd.OutputStream(channels=1, samplerate=16000, dtype='int16')
    stream.start()
    try:
        while True:
            audio_frame = (yield)
            data = np.frombuffer(audio_frame.data, dtype=np.int16)
            # stream.write(data)
    finally:
        stream.stop()
        stream.close()


@coroutine
def vu_meter_co():
    max_rms = 1.0
    num_cols = 70
    count = 0
    if FLAGS.show_ffmpeg_output:
        return
    while True:
        max_rms = max(max_rms * 0.99, 1.0)
        audio_frame = (yield)
        data = audio_frame.data
        ts = audio_frame.timestamp
        rms = audioop.rms(data, 2)
        try:
            norm_rms = rms / len(data)
        except ZeroDivisionError:
            continue
        if norm_rms > max_rms:
            max_rms = norm_rms
        bar_cols = int(num_cols * (norm_rms / max_rms))
        sys.stdout.write('\r[' + ('*' * bar_cols) +
                         (' ' * (num_cols - bar_cols)) +
                         '] %4.1f %s' % (norm_rms, count))
        count += 1
        sys.stdout.flush()


@coroutine
def reframer_co(target, desired_frame_duration_ms=None):
    sample_rate = None
    data = bytes()
    ts = None
    total_bytes = 0
    while True:
        frame = (yield)
        if not sample_rate:
            sample_rate = frame.sample_rate
            num_bytes = 2 * sample_rate * desired_frame_duration_ms // 1000
        if not ts:
            ts = frame.timestamp
        data += frame.data
        while len(data) > num_bytes:
            buf = data[:num_bytes]
            total_bytes += num_bytes
            frame = _AudioFrame(
                data=buf, timestamp=ts, sample_rate=sample_rate)
            if target:
                target.send(frame)
            ts += timedelta(milliseconds=desired_frame_duration_ms)
            data = data[num_bytes:]

@coroutine
def vad_trigger_co(target, sample_rate=None, frame_duration_ms=None,
                   padding_duration_ms=None):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    triggered_ts = None
    vad = webrtcvad.Vad()
    while True:
        frame = (yield)
        is_speech = vad.is_speech(frame.data, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    if not triggered_ts:
                        triggered_ts = f.timestamp
                target.send(('triggered', triggered_ts))
                for f, s in ring_buffer:
                    target.send(('audio', f))
                ring_buffer.clear()
        else:
            target.send(('audio', frame))
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                target.send(('detriggered', frame.timestamp))
                triggered = False
                triggered_ts = None
                ring_buffer.clear()


@coroutine
def vad_collector_co(utterance_queue=None):
    frames = []
    while True:
        event, data = (yield)
        if event == 'triggered':
            frames = []
        elif event == 'detriggered':
            utterance_queue.put(_AudioFrame.coalesce(frames))
        else:
            frames.append(data)


def enqueue_stream_input(f, queue, bufsiz):
    while True:
        ts = datetime.now()
        data = f.read(bufsiz)
        queue.put((ts, data))


def start_queueing_stream_input(f, queue, bufsiz):
    start_daemon(enqueue_stream_input, f, queue, bufsiz)


@coroutine
def print_co():
    while True:
        print((yield))

def coalesce_stream_input_queue(q):
    data = b''
    max_count = 100
    count = 0
    while not q.empty() and count < max_count:
        ts, buf = q.get()
        data += buf
        count += 1
    return data


def process_utterance_queue(queue):
    recognizer = _BingRecognizer()
    while True:
        audio = queue.get()
        wav_filename = '%s.wav' % (timestamp_str(audio.timestamp))
        wav_path = os.path.join(FLAGS.wav_output_dir, wav_filename)
        wav_file = wave.open(wav_path, 'wb')
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio.data)
        wav_file.close()
        result = recognizer.recognize_file(wav_path)
        print_recog_result(audio.timestamp, result)
        q.put([audio.timestamp, result])


def print_recog_result(ts, result):
    print_status('%s %s', ts, termcolor.colored(result, attrs=['bold']))


class AudioListener:
    def __init__(self, url):
        try:
            shutil.rmtree(FLAGS.wav_output_dir)
            os.makedirs(FLAGS.wav_output_dir)
        except FileExistsError:
            pass
        ffmpeg_cmd = ffmpeg.input(url).output(
            '-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k').compile()
        print(ffmpeg_cmd)
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        ffmpeg_stdout_q = queue.Queue()
        ffmpeg_stderr_q = queue.Queue()
        start_queueing_stream_input(ffmpeg_proc.stdout, ffmpeg_stdout_q, 1600)
        start_queueing_stream_input(ffmpeg_proc.stderr, ffmpeg_stderr_q, 1)
        utterance_queue = queue.Queue()
        start_daemon(process_utterance_queue, utterance_queue)
        start_cleanup(remove_wav)
        play_audio = play_audio_co()
        vu_meter = vu_meter_co()
        vad_collector = vad_collector_co(utterance_queue=utterance_queue)
        vad_trigger = vad_trigger_co(
            vad_collector,
            sample_rate=16000, frame_duration_ms=30, padding_duration_ms=500)
        reframer = reframer_co(vad_trigger, desired_frame_duration_ms=30)
        audio_pipeline_head = broadcast([reframer, vu_meter, play_audio])
        while True:
            got_item = False
            if not ffmpeg_stderr_q.empty():
                got_item = True
                stderr_buf = coalesce_stream_input_queue(ffmpeg_stderr_q)
                if FLAGS.show_ffmpeg_output:
                    sys.stderr.write(stderr_buf.decode('utf8'))
                    sys.stderr.flush()
            audio_item = queue_get_nowait(ffmpeg_stdout_q)
            if audio_item:
                got_item = True
                ts, audio_buf = audio_item
                audio_frame = _AudioFrame(data=audio_buf, timestamp=ts, sample_rate=16000)
                audio_pipeline_head.send(audio_frame)
            if not got_item:
                time.sleep(0)
