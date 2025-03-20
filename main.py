#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "librosa>=0.11.0",
#     "numpy>=2.1.3",
#     "pyrubberband>=0.4.0",
#     "scipy>=1.15.2",
#     "soundfile>=0.13.1",
# ]
# ///
import argparse
import logging
import tempfile
from enum import StrEnum
from typing import BinaryIO

import librosa
import numpy as np
import pyrubberband as pyrb
import soundfile as sf
from librosa import feature
from scipy import signal


logger = logging.getLogger('voicechange')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Method(StrEnum):
    """Enum for the method of voice change."""
    pitch = 'pitch'
    formant = 'formant'
    vocoder = 'vocoder'
    combined = 'combined'


def pitch(y: np.ndarray, sr: int | float) -> np.ndarray:
    pitch_shift = -2
    y_shifted = pyrb.pitch_shift(y, sr, pitch_shift)

    rate = 0.95  # slow down 5%
    return pyrb.time_stretch(y_shifted, sr, rate)


def formant(y: np.ndarray) -> np.ndarray:
    formant_shift = 1.3  # increase formant by 30%

    s = np.abs(librosa.stft(y))

    num_freqs = s.shape[0]
    new_s = np.zeros_like(s)

    for i in range(s.shape[1]):
        stretched = signal.resample(s[:, i], int(num_freqs / formant_shift))

        if stretched.shape[0] < num_freqs:
            new_s[:stretched.shape[0], i] = stretched
        else:
            new_s[:, i] = stretched[:num_freqs]

    # resampling
    y_complex = librosa.stft(y)
    phase = np.angle(y_complex)

    return librosa.istft(new_s * np.exp(1j * phase))


def vocoder(y: np.ndarray, sr: int | float) -> np.ndarray:
    fmin = float(librosa.note_to_hz('C2'))
    fmax = float(librosa.note_to_hz('C7'))

    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin, fmax=fmax)

    f0_shifted = np.zeros_like(f0)
    for i in range(len(f0)):
        if np.isnan(f0[i]):
            f0_shifted[i] = np.nan
        else:
            # no linear tone shift
            f0_shifted[i] = f0[i] * (1.2 - 0.1 * np.sin(i / 100))

    # MFCC coefficients
    mfcc = feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # change MFCC (change timbre)
    mfcc[1:, :] = mfcc[1:, :] * 1.3  # up

    # restore MFCC
    y_recon = feature.inverse.mfcc_to_audio(mfcc, n_fft=2048, hop_length=512)

    return pyrb.pitch_shift(y_recon, sr, 2)


def combined(y: np.ndarray, sr: int | float, file_ext: str) -> np.ndarray:
    # change pitch
    y_shifted = pyrb.pitch_shift(y, sr, -3)

    with tempfile.NamedTemporaryFile(delete_on_close=False, suffix=f'.{file_ext}') as temp:
        sf.write(temp, y_shifted, sr)
        temp.close()

        # change frequency
        y_formant, _ = librosa.load(temp.name, sr=int(sr * 1.2))

    # add noise
    noise = np.random.randn(len(y_formant)) * 0.005
    y_noised = y_formant + noise

    # timbre filter
    b, a = [0.9, -0.3, 0.1], [1.0, -0.2, 0.1]
    y_filtered = signal.lfilter(b, a, y_noised)

    # charge speed
    y_final = pyrb.time_stretch(y_filtered, sr, 0.95)

    # normalize volume
    # y_final / np.max(np.abs(y_final))
    return librosa.util.normalize(y_final)


def main(input_file: BinaryIO, output_file: BinaryIO, method: Method) -> None:
    file_ext = input_file.name.rsplit('.', 1)[-1]
    logger.info('start processing ext=%s with mode=%s', file_ext, method)

    y, sr = librosa.load(input_file, sr=None)

    match method:
        case Method.pitch:
            processed_audio = pitch(y, sr)
        case Method.formant:
            processed_audio = formant(y)
        case Method.vocoder:
            processed_audio = vocoder(y, sr)
        case _:
            processed_audio = combined(y, sr, file_ext)

    logger.info('finish processing, length=%s', len(processed_audio))
    sf.write(output_file, processed_audio, sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voice change tool')
    parser.add_argument(
        '-f', '--file',
        dest='file',
        type=argparse.FileType(mode='rb'),
        required=True,
        help='audio file',
    )
    parser.add_argument(
        '-o', '--output',
        dest='output',
        required=True,
        type=argparse.FileType(mode='wb'),
        help='output file',
    )
    parser.add_argument('-m', '--method', dest='method', type=Method, help='method', default=Method.combined)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose mode')

    args, _ = parser.parse_known_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    main(args.file, args.output, args.method)
