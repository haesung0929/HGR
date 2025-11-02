
import time
from threading import Thread, Event
from queue import Queue, Empty

class TTSWorker(Thread):
    """
    Threaded TTS worker (Windows-friendly).
    - Tries win32com SAPI first; falls back to pyttsx3.
    - Call speak(text) to enqueue speech.
    - Call shutdown() to stop the thread gracefully.
    """
    def __init__(self, rate=0, volume=1.0):
        super().__init__(daemon=True)
        self.queue = Queue()
        self.stop_event = Event()
        self.rate = rate            # -10..+10 for SAPI, or typical wpm for pyttsx3
        self.volume = volume        # 0.0..1.0
        self.use_win32 = None

    def speak(self, text: str):
        if text is None:
            return
        text = str(text).strip()
        if not text:
            return
        self.queue.put(text)

    def run(self):
        # Initialize COM (required for win32com usage)
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_inited = True
        except Exception:
            com_inited = False

        engine = None
        speaker = None
        try:
            # Prefer Windows SAPI (stable & fast on Windows)
            try:
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                if self.rate != 0:
                    try:
                        speaker.Rate = self.rate
                    except Exception:
                        pass
                try:
                    speaker.Volume = int(self.volume * 100)
                except Exception:
                    pass
                self.use_win32 = True
            except Exception:
                # Fallback: pyttsx3
                import pyttsx3
                try:
                    engine = pyttsx3.init(driverName='sapi5')
                except Exception:
                    engine = pyttsx3.init()
                if self.rate != 0:
                    try:
                        # choose a reasonable default rate if requested
                        engine.setProperty('rate', 180)
                    except Exception:
                        pass
                try:
                    engine.setProperty('volume', self.volume)
                except Exception:
                    pass
                self.use_win32 = False

            while not self.stop_event.is_set():
                try:
                    txt = self.queue.get(timeout=0.1)
                except Empty:
                    continue

                try:
                    if self.use_win32 and speaker is not None:
                        # Speak synchronously
                        speaker.Speak(txt)
                    else:
                        # pyttsx3
                        engine.stop()  # clear any pending
                        engine.say(txt)
                        engine.runAndWait()
                except Exception as e:
                    print("TTS error:", e)

        finally:
            if com_inited:
                try:
                    import pythoncom
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def shutdown(self):
        self.stop_event.set()
        # Wake up the queue to allow thread to exit promptly
        self.queue.put("")

# Simple time-based anti-spam helper
class SpeakGate:
    def __init__(self, cooldown=0.7):
        self.cooldown = cooldown
        self._last_text = None
        self._last_ts = 0.0

    def can_speak(self, text, now=None):
        if now is None:
            now = time.time()
        if text == self._last_text and (now - self._last_ts) < (self.cooldown * 2):
            return False
        if (now - self._last_ts) < self.cooldown:
            return False
        self._last_text = text
        self._last_ts = now
        return True
