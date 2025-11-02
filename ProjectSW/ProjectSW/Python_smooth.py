
import time
import collections
import cv2
import mediapipe as mp

from tts_worker import TTSWorker, SpeakGate

# ------------------- Parameters you might tweak -------------------
MAX_HANDS = 2
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5
BUFFER_SIZE = 5           # sliding window per hand
STABLE_THRESHOLD = 3      # majority needed to accept a letter
TTS_COOLDOWN = 0.7        # seconds between speeches (anti-spam)
FONT_SCALE = 1.4
THICKNESS = 2

# ------------------- Init MediaPipe Hands -------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
    model_complexity=1
)

# ------------------- Init TTS -------------------
tts = TTSWorker(rate=0, volume=1.0)
tts.start()
gate = SpeakGate(cooldown=TTS_COOLDOWN)

# ------------------- Helpers -------------------
def most_common_nonempty(buffer):
    counter = collections.Counter([x for x in buffer if x])
    if not counter:
        return ''
    return counter.most_common(1)[0][0]

def get_alphabet(myHand):
    """
    Reuse the user's simple rule-set from original Python.py.
    myHand: list of [x, y] (pixel coords) for 21 landmarks (0..20)
    Return: detected single uppercase letter or '' if unknown.
    """
    try:
        # shorthand
        mh = myHand
        alphabet_config = {
            'A': [mh[4][0] > mh[2][0],  mh[8][1] > mh[5][1],  mh[12][1] > mh[9][1],
                  mh[16][1] > mh[13][1], mh[20][1] > mh[17][1]],
            'B': [mh[4][0] < mh[2][0],  mh[8][1] < mh[5][1],  mh[12][1] < mh[9][1],
                  mh[16][1] < mh[13][1], mh[20][1] < mh[17][1]],
            'C': [mh[4][0] > mh[5][0],  mh[8][0] > mh[5][0],  mh[12][0] > mh[9][0],
                  mh[16][0] > mh[13][0], mh[20][1] > mh[18][1]],
            'D': [mh[4][0] > mh[12][0], mh[8][0] > mh[5][0],  mh[12][0] > mh[9][0],
                  mh[16][0] > mh[13][0], mh[20][1] < mh[18][1]],
            'E': [mh[4][0] < mh[11][0], mh[8][1] > mh[5][1],  mh[12][1] > mh[9][1],
                  mh[16][1] > mh[13][1], mh[20][1] > mh[17][1]],
            'F': [mh[4][0] > mh[2][0],  mh[8][1] > mh[5][1],  mh[12][1] < mh[9][1],
                  mh[16][1] < mh[13][1], mh[20][1] < mh[17][1]],
            'I': [mh[4][0] < mh[10][0], mh[8][1] > mh[5][1],  mh[12][1] > mh[9][1],
                  mh[16][1] > mh[13][1], mh[20][1] < mh[17][1]],
            'K': [mh[4][0] > mh[2][0],  mh[8][1] < mh[5][1],  mh[12][1] < mh[9][1],
                  mh[16][1] > mh[13][1], mh[20][1] > mh[17][1]],
            'L': [mh[4][0] > mh[2][0],  mh[4][0] > mh[12][0], mh[8][1] < mh[5][1],
                  mh[12][1] > mh[9][1], mh[16][1] > mh[13][1], mh[20][1] > mh[17][1]],
            'P': [mh[4][0] > mh[11][0], mh[8][0] < mh[6][0],  mh[12][0] < mh[9][0],
                  mh[16][0] < mh[13][0], mh[20][0] < mh[17][0]],
            'S': [mh[4][0] > mh[11][0], mh[8][1] > mh[5][1],  mh[12][1] > mh[9][1],
                  mh[16][1] > mh[13][1], mh[20][1] > mh[17][1]],
            'V': [mh[4][0] < mh[2][0],  mh[8][1] < mh[5][1],  mh[12][1] < mh[9][1],
                  mh[16][1] > mh[13][1], mh[20][1] > mh[17][1]],
            'W': [mh[4][0] < mh[2][0],  mh[8][1] < mh[5][1],  mh[12][1] < mh[9][1],
                  mh[16][1] < mh[13][1], mh[20][1] > mh[17][1]],
            'Y': [mh[4][0] > mh[10][0], mh[8][1] > mh[5][1],  mh[12][1] > mh[9][1],
                  mh[16][1] > mh[13][1], mh[20][1] < mh[17][1]],
        }

        for letter, conds in alphabet_config.items():
            if all(conds):
                return letter
        return ''
    except Exception:
        return ''

def draw_label(img, text, org, color=(0, 0, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

# ------------------- Camera -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    raise SystemExit

last_stable = ['','']                         # stable accepted letter per hand
buffers = [collections.deque(maxlen=BUFFER_SIZE) for _ in range(MAX_HANDS)]
fps_ts = time.time()
fps = 0.0

try:
    while True:
        ok, img = cap.read()
        if not ok:
            print("Error: Unable to read frame from camera.")
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # FPS calc
        now = time.time()
        dt = now - fps_ts
        if dt > 0:
            fps = (0.9 * fps) + (0.1 * (1.0 / dt))
        fps_ts = now
        draw_label(img, f"FPS: {fps:.1f}", (10, 30), (0, 255, 0))

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
                # Extract pixel landmarks
                h, w, _ = img.shape
                myHand = []
                for lm in hand_landmarks.landmark:
                    myHand.append([int(lm.x * w), int(lm.y * h)])

                # Basic visualization
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Instant (raw) detection
                raw_letter = get_alphabet(myHand)

                # Update buffer & compute stable letter
                buffers[hand_idx].append(raw_letter)
                stable_letter = most_common_nonempty(buffers[hand_idx])

                # If letter stabilized & changed, speak it
                if stable_letter and stable_letter != last_stable[hand_idx]:
                    if gate.can_speak(stable_letter, now=time.time()):
                        tts.speak(stable_letter)
                    last_stable[hand_idx] = stable_letter

                # Draw raw & stable letters near wrist (landmark 0)
                if myHand:
                    base_x, base_y = myHand[0][0], myHand[0][1]
                    if stable_letter:
                        draw_label(img, f"{stable_letter}", (base_x, max(30, base_y - 10)))
                    elif raw_letter:
                        draw_label(img, f"? {raw_letter}", (base_x, max(30, base_y - 10)), (255, 255, 0))

        else:
            draw_label(img, "No hand detected", (10, 70), (0, 0, 255))

        cv2.imshow("Smooth Alphabet Recognition (per-change TTS)", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    tts.shutdown()
