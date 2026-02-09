"""
Apple-native audio conversion (Apple Silicon macOS only).

Converts arbitrary audio files to mono 16-bit PCM WAV at a given sample rate
using Apple tools: AVFoundation when input already has the target sample rate,
otherwise the system afconvert CLI. No UI or app dependencies.

When this module returns a path to a newly created temp file, the caller is
responsible for deleting that file when done.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import wave

# macOS-only: AVFoundation (and Foundation for NSURL) used when SR matches.
_AVFoundation = None
_Foundation = None

if sys.platform == "darwin":
    try:
        import Foundation as _Foundation
        import AVFoundation as _AVFoundation
    except Exception:
        pass


def convert_to_wav(
    input_path: str,
    sample_rate: int = 24000,
) -> str | None:
    """Convert an audio file to mono 16-bit PCM WAV at the given sample rate.

    Uses Apple's AVFoundation on macOS when possible; falls back to the
    system afconvert command when sample-rate conversion is needed. On other
    platforms, returns None.

    - If the file is already a WAV with 1 channel and the target sample rate,
      returns input_path unchanged.
    - Otherwise, converts to a new temp WAV and returns its path. The caller
      must delete that temp file when finished.
    - Returns None if the file does not exist, conversion is not supported
      (e.g. not on Apple Silicon macOS), or conversion fails.

    This module does not print or log; the caller may do so.
    """
    if not os.path.exists(input_path):
        return None

    _, ext = os.path.splitext(input_path)
    if ext.lower() == ".wav":
        try:
            with wave.open(input_path, "rb") as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == sample_rate:
                    return input_path
        except wave.Error:
            pass

    if sys.platform != "darwin":
        return None

    temp_file = tempfile.NamedTemporaryFile(
        prefix="qwen3_convert_",
        suffix=".wav",
        delete=False,
    )
    temp_wav = temp_file.name
    temp_file.close()

    try:
        # For non-WAV, use afconvert only (AVFoundation can crash on some m4a/opus).
        # For WAV that needs conversion, try AVFoundation when SR matches, else afconvert.
        if ext.lower() != ".wav":
            _convert_with_afconvert(input_path, temp_wav, sample_rate)
        elif _AVFoundation is not None and _Foundation is not None:
            _convert_with_avfoundation_or_afconvert(input_path, temp_wav, sample_rate)
        else:
            _convert_with_afconvert(input_path, temp_wav, sample_rate)
        return temp_wav
    except Exception:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass
        return None


def _convert_with_afconvert(input_path: str, output_path: str, sample_rate: int) -> None:
    """Convert using macOS afconvert CLI. Raises on failure."""
    # -f WAVE = WAV, -d LEI16 = little-endian 16-bit int, @sample_rate, -c 1 = mono
    cmd = [
        "afconvert",
        "-f", "WAVE",
        "-d", f"LEI16@{sample_rate}",
        "-c", "1",
        input_path,
        output_path,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"afconvert failed: {result.stderr or result.stdout or result.returncode}")


def _convert_with_avfoundation_or_afconvert(
    input_path: str,
    output_path: str,
    sample_rate: int,
) -> None:
    """Use AVFoundation if input sample rate matches target; else afconvert."""
    Foundation = _Foundation
    AVFoundation = _AVFoundation

    in_url = Foundation.NSURL.fileURLWithPath_(input_path)
    in_file, err = AVFoundation.AVAudioFile.alloc().initForReading_error_(in_url, None)
    if in_file is None or err is not None:
        raise RuntimeError(f"Failed to open input: {err}")

    in_format = in_file.processingFormat()
    in_sr = int(in_format.sampleRate())

    if in_sr == sample_rate:
        # Same sample rate: convert format only (e.g. Float32 -> Int16) with AVFoundation
        _convert_same_rate_avfoundation(in_file, in_format, output_path, sample_rate)
    else:
        # Different sample rate: use afconvert (AVAudioConverter SRC fails with -50 in PyObjC)
        _convert_with_afconvert(input_path, output_path, sample_rate)


def _convert_same_rate_avfoundation(
    in_file,
    in_format,
    output_path: str,
    sample_rate: int,
) -> None:
    """Convert at same sample rate (format only) using AVAudioConverter. Raises on failure."""
    AVFoundation = _AVFoundation
    Foundation = _Foundation

    # Output format: 16-bit PCM, mono, target sample rate
    pcm_int16 = 3
    out_format = AVFoundation.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
        pcm_int16,
        float(sample_rate),
        1,
        True,
    )
    if out_format is None:
        raise RuntimeError("Failed to create output format")

    converter = AVFoundation.AVAudioConverter.alloc().initFromFormat_toFormat_(
        in_format,
        out_format,
    )
    if converter is None:
        raise RuntimeError("Failed to create converter")

    out_url = Foundation.NSURL.fileURLWithPath_(output_path)
    out_file, err = AVFoundation.AVAudioFile.alloc().initForWriting_settings_error_(
        out_url,
        out_format.settings(),
        None,
    )
    if out_file is None or err is not None:
        raise RuntimeError(f"Failed to create output file: {err}")

    frame_capacity = 4096
    in_buffer = AVFoundation.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
        in_format,
        frame_capacity,
    )
    out_buffer = AVFoundation.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
        out_format,
        frame_capacity,
    )

    while True:
        ok, err = in_file.readIntoBuffer_error_(in_buffer, None)
        if not ok or err is not None:
            break
        frame_count = in_buffer.frameLength()
        if frame_count == 0:
            break

        out_buffer.setFrameLength_(0)
        conv_ok, conv_err = converter.convertToBuffer_fromBuffer_error_(
            out_buffer,
            in_buffer,
            None,
        )
        if conv_err is not None:
            raise RuntimeError(f"Conversion error: {conv_err}")
        out_frames = out_buffer.frameLength()
        if out_frames > 0:
            out_file.writeFromBuffer_error_(out_buffer, None)
