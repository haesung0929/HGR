# ex-python-twohands.py
# -------------------------------------------------------
# MediaPipe로 두 손(0..10)의 손가락 개수 세기:
# - 엄지는 Left/Right(손 우·좌) 방향으로 판정
# - 핀치(OK)는 손 크기 비율 기반(스케일 불변)
# - 두 손의 손가락 수 합산 → 총 0..10
# - TTS 백그라운드 실행 + 명령 합치기 + 쿨다운
# - 총합이 N 프레임 연속으로 안정될 때만 재생(디바운스)
# - 손마다 개수 + 총합, 그리고 핀치 시 "OK" 표시
# -------------------------------------------------------

import time
import math
import cv2
import mediapipe as mp
from threading import Thread, Event
from queue import Queue, Empty

# ========================= TTS 워커 =========================
class TTSWorker(Thread):
    def __init__(self, rate=0, volume=1.0):
        super().__init__(daemon=True)
        self.queue = Queue()
        self.stop_event = Event()
        self.rate = rate
        self.volume = volume
        self.engine_kind = None   # 엔진 종류: "win32" | "pyttsx3"

    def speak(self, text: str):
        if text is None:
            return
        self.queue.put(str(text))

    def shutdown(self):
        self.stop_event.set()
        self.queue.put(None)  # 센티넬

    def _init_win32(self):
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            try: speaker.Rate = int(self.rate)
            except Exception: pass
            try:
                vol = int(max(0, min(100, int(self.volume * 100))))
                speaker.Volume = vol
            except Exception: pass
            return speaker
        except Exception:
            return None

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            try:
                base_rate = engine.getProperty("rate") or 200
                engine.setProperty("rate", base_rate + int(self.rate) * 10)
            except Exception: pass
            try:
                engine.setProperty("volume", float(max(0.0, min(1.0, self.volume))))
            except Exception: pass
            return engine
        except Exception:
            return None

    def run(self):
        speaker = self._init_win32()
        if speaker is not None:
            self.engine_kind = "win32"
        else:
            engine = self._init_pyttsx3()
            if engine is None:
                self.engine_kind = None
                return
            self.engine_kind = "pyttsx3"

        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if item is None:
                break

            # 합치기: 마지막 요소만 유지
            text = item
            try:
                while True:
                    nxt = self.queue.get_nowait()
                    if nxt is None:
                        text = None
                        self.queue.put(None)
                        break
                    text = nxt
            except Empty:
                pass

            if not text:
                continue

            try:
                if self.engine_kind == "win32":
                    # 1=비동기, 2=퍼지 → 3=이전 발화 중단 & 즉시 말하기
                    speaker.Speak(text, 3)
                else:
                    engine.stop()
                    engine.say(text)
                    engine.runAndWait()
            except Exception:
                if self.engine_kind == "win32":
                    engine = self._init_pyttsx3()
                    if engine is not None:
                        self.engine_kind = "pyttsx3"
                    else:
                        break

# ========================= MEDIAPIPE 설정 =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
try:
    mp_styles = mp.solutions.drawing_styles
except Exception:
    mp_styles = None

# ========================= 유틸리티 =========================
last_spoken_text = None
last_spoken_ts = 0.0
SPEAK_COOLDOWN = 0.9          # 초(두 발화 사이 최소 간격)
STABILITY_FRAMES = 4          # 말하기 전 연속 안정 프레임 수

def can_speak(text: str, now: float) -> bool:
    global last_spoken_text, last_spoken_ts
    if text == last_spoken_text and (now - last_spoken_ts) < (SPEAK_COOLDOWN * 2.0):
        return False
    if (now - last_spoken_ts) < SPEAK_COOLDOWN:
        return False
    last_spoken_text = text
    last_spoken_ts = now
    return True

def bbox_area(hand_landmarks) -> float:
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return max(0.0, (max(xs) - min(xs))) * max(0.0, (max(ys) - min(ys)))

def hand_bbox_px(hand_landmarks, frame_shape):
    h, w = frame_shape[:2]
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
    return min(xs), min(ys), max(xs), max(ys)

def count_fingers(hand_landmarks, hand_label: str) -> int:
    fc = 0
    # 엄지: X축 + 손 좌/우(이미지 좌우 반전 적용됨)
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ip  = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    if (hand_label == "Right" and tip.x < ip.x) or (hand_label == "Left" and tip.x > ip.x):
        fc += 1

    pairs = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP,  mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP,   mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP,         mp_hands.HandLandmark.PINKY_PIP),
    ]
    for tip_id, pip_id in pairs:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y:
            fc += 1
    return fc

def pinch_ratio(hand_landmarks) -> float:
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    mcp   = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    scale = math.hypot(wrist.x - mcp.x, wrist.y - mcp.y) + 1e-6
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return dist / scale

# ========================= 메인 =========================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tts = TTSWorker(rate=0, volume=1.0)
    tts.start()

    # TTS 발화를 위한 안정 상태
    prev_for_stable = None
    same_frames = 0
    last_spoken_total = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Không đọc được khung hình từ camera.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                total_count = 0
                per_hand_counts = {"Left": None, "Right": None}

                if results.multi_hand_landmarks:
                    labels = []
                    if results.multi_handedness:
                        labels = [h.classification[0].label for h in results.multi_handedness]
                    else:
                        labels = ["Right"] * len(results.multi_hand_landmarks)

                    # 각 손에 대해 그리기/계산
                    for hand_lms, hand_label in zip(results.multi_hand_landmarks, labels):
                        # 랜드마크 그리기
                        if mp_styles is not None:
                            mp_drawing.draw_landmarks(
                                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                mp_styles.get_default_hand_landmarks_style(),
                                mp_styles.get_default_hand_connections_style()
                            )
                        else:
                            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                        # 해당 손의 손가락 수 세기
                        fc = count_fingers(hand_lms, hand_label)
                        total_count += fc
                        per_hand_counts[hand_label] = fc

                        # 핀치 오버레이(TTS는 방해 방지를 위해 재생 안 함)
                        if pinch_ratio(hand_lms) < 0.7:
                            x1, y1, x2, y2 = hand_bbox_px(hand_lms, frame.shape)
                            cv2.putText(frame, "OK", (x1, max(30, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        # 손의 bbox 근처에 손가락 개수 라벨 표시
                        x1, y1, x2, y2 = hand_bbox_px(hand_lms, frame.shape)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                        cv2.putText(frame, f"{hand_label}: {fc}", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # 총합 표시(0..10)
                    total_txt = f"TOTAL: {total_count}"
                    cv2.putText(frame, total_txt, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    # 좌/우 값을 화면 구석에 표시
                    lr_txt = f"L: {per_hand_counts['Left'] if per_hand_counts['Left'] is not None else '-'} | " \
                             f"R: {per_hand_counts['Right'] if per_hand_counts['Right'] is not None else '-'}"
                    cv2.putText(frame, lr_txt, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # TTS 발화 디바운스: 총합이 STABILITY_FRAMES 프레임 동안 안정될 때만 말함
                    if prev_for_stable is None or total_count != prev_for_stable:
                        prev_for_stable = total_count
                        same_frames = 1
                    else:
                        same_frames += 1

                    if same_frames >= STABILITY_FRAMES and total_count != last_spoken_total:
                        now = time.time()
                        if can_speak(str(total_count), now):
                            tts.speak(str(total_count))
                            last_spoken_total = total_count

                else:
                    cv2.putText(frame, "Gesture not detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # 손이 사라지면 안정 상태 리셋
                    prev_for_stable = None
                    same_frames = 0
                    time.sleep(0.001)  # CPU 부하 감소

                cv2.imshow("Two-Hands Finger Counter (0..10)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            tts.shutdown()
            try:
                tts.join(timeout=1.0)
            except Exception:
                pass

if __name__ == "__main__":
    main()
