"""Apple Speech (macOS built-in) transcription. Apple Silicon macOS only (darwin/arm64), optional."""

import os
import sys
import time

APPLE_SPEECH_AVAILABLE = False
Foundation = None
SFSpeechRecognizer = None
SFSpeechURLRecognitionRequest = None

if sys.platform == "darwin":
    try:
        import Foundation as _Foundation
        from Speech import (
            SFSpeechRecognizer as _SFSpeechRecognizer,
            SFSpeechURLRecognitionRequest as _SFSpeechURLRecognitionRequest,
        )
        Foundation = _Foundation
        SFSpeechRecognizer = _SFSpeechRecognizer
        SFSpeechURLRecognitionRequest = _SFSpeechURLRecognitionRequest
        APPLE_SPEECH_AVAILABLE = True
    except Exception:
        pass


def transcribe_wav_with_apple_speech(wav_path: str) -> str | None:
    """Transcribe a WAV file using macOS built-in Speech framework.

    Returns the transcribed text, or None if unavailable, denied, or failed.
    """
    if not APPLE_SPEECH_AVAILABLE or Foundation is None or not os.path.exists(wav_path):
        return None

    AUTHORIZED = 3
    auth_result = [None]

    def auth_callback(status):
        auth_result[0] = status

    SFSpeechRecognizer.requestAuthorization_(auth_callback)
    run_loop = Foundation.NSRunLoop.currentRunLoop()
    deadline = time.time() + 60
    while auth_result[0] is None and time.time() < deadline:
        run_loop.runUntilDate_(
            Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )
    if auth_result[0] != AUTHORIZED:
        return None

    locale = Foundation.NSLocale.localeWithLocaleIdentifier_("en-US")
    recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale)
    if recognizer is None or not recognizer.isAvailable():
        return None

    url = Foundation.NSURL.fileURLWithPath_(wav_path)
    request = SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
    if request is None:
        return None

    transcript_result = [None]
    transcript_error = [None]

    def result_handler(result, error):
        if error is not None:
            transcript_error[0] = error
            return
        if result is not None and result.isFinal():
            trans = result.bestTranscription()
            if trans is not None:
                transcript_result[0] = trans.formattedString()

    task = recognizer.recognitionTaskWithRequest_resultHandler_(request, result_handler)
    if task is None:
        return None

    deadline = time.time() + 60
    while transcript_result[0] is None and transcript_error[0] is None and time.time() < deadline:
        run_loop.runUntilDate_(
            Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )

    text = transcript_result[0]
    return (text.strip() or None) if text else None


def _offer_apple_transcribe(wav_path: str, prompt_msg: str) -> str | None:
    """If Apple Speech is available, ask user and transcribe WAV. Returns transcript or None."""
    from qwen3_tts.ui import console, instant_menu_choice

    if not APPLE_SPEECH_AVAILABLE:
        return None
    choice = instant_menu_choice(
        f"  [accent]{prompt_msg}[/accent] [muted](y/n)[/muted]: ",
        {"y", "n"},
    )
    if choice != "y":
        return None
    console.print("  [info]Transcribing with Apple...[/info]")
    result = transcribe_wav_with_apple_speech(wav_path)
    if result:
        console.print(f"  [success]\u2713 Transcript:[/success] [muted]{result[:80]}{'...' if len(result) > 80 else ''}[/muted]")
    else:
        console.print("  [warning]Transcription unavailable or denied.[/warning]")
    return result
