
import time
import math
import collections
import cv2
import mediapipe as mp

from tts_worker import TTSWorker, SpeakGate  # TTS worker + anti-spam gate

# =================== Config ===================
MAX_HANDS = 2
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5
MODEL_COMPLEXITY = 1

BUFFER_SIZE = 5           # sliding window for smoothing
STABLE_THRESHOLD = 3      # majority threshold to accept a decision
TTS_COOLDOWN = 0.7        # seconds between spoken letters
FONT_SCALE = 1.2
THICKNESS = 2

OK_RATIO_THRESH = 0.23    # OK pinch threshold (normalized distance / hand box diag)

# Behavior toggles
SPEAK_OK = False          # Only speak letters; keep False to silence "OK"
SHOW_OK = True            # Draw "OK" label when stable
SPEAK_LETTERS = True      # Speak letters when they change

# =================== MediaPipe ===================
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
    model_complexity=MODEL_COMPLEXITY
)

# =================== TTS ===================
tts = TTSWorker(rate=0, volume=1.0)
tts.start()
gate = SpeakGate(cooldown=TTS_COOLDOWN)

# =================== Helpers ===================
def draw_label(img, text, org, color=(0, 0, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

def most_common_nonempty(buffer):
    counter = collections.Counter([x for x in buffer if x])
    if not counter:
        return ''
    return counter.most_common(1)[0][0]

def is_ok_pinch(hand_landmarks):
    """
    OK detection by normalized distance between thumb tip and index tip,
    scaled by hand bounding-box diagonal (robust over distance to camera).
    """
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        dx = thumb_tip.x - index_tip.x
        dy = thumb_tip.y - index_tip.y
        dist = math.hypot(dx, dy)

        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if diag < 1e-6:
            return False
        ratio = dist / diag
        return ratio < OK_RATIO_THRESH
    except Exception:
        return False


def _angle(ax, ay, bx, by, cx, cy):
    """Return angle ABC in degrees given three points A, B, C."""
    import math
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot = v1x * v2x + v1y * v2y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))

def _hand_diag(lm):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    return math.hypot(max(xs)-min(xs), max(ys)-min(ys))

def count_fingers(hand_landmarks):
    """
    Angle-based finger extension:
      - Index/Middle/Ring/Pinky: extended if PIP and DIP angles > 160°.
      - Thumb: extended if MCP and IP angles > 160° AND
               tip far enough from index MCP relative to hand size.
    More robust to rotation and reduces false positives when thumb is folded.
    """
    H = mp_hands.HandLandmark
    lm = hand_landmarks.landmark

    diag = _hand_diag(lm)
    if diag < 1e-6:
        return 0

    def ang(a, b, c):
        return _angle(lm[a].x, lm[a].y, lm[b].x, lm[b].y, lm[c].x, lm[c].y)

    cnt = 0

    # Index
    pip = ang(H.INDEX_FINGER_MCP, H.INDEX_FINGER_PIP, H.INDEX_FINGER_DIP)
    dip = ang(H.INDEX_FINGER_PIP, H.INDEX_FINGER_DIP, H.INDEX_FINGER_TIP)
    if pip > 160 and dip > 160:
        cnt += 1

    # Middle
    pip = ang(H.MIDDLE_FINGER_MCP, H.MIDDLE_FINGER_PIP, H.MIDDLE_FINGER_DIP)
    dip = ang(H.MIDDLE_FINGER_PIP, H.MIDDLE_FINGER_DIP, H.MIDDLE_FINGER_TIP)
    if pip > 160 and dip > 160:
        cnt += 1

    # Ring
    pip = ang(H.RING_FINGER_MCP, H.RING_FINGER_PIP, H.RING_FINGER_DIP)
    dip = ang(H.RING_FINGER_PIP, H.RING_FINGER_DIP, H.RING_FINGER_TIP)
    if pip > 160 and dip > 160:
        cnt += 1

    # Pinky
    pip = ang(H.PINKY_MCP, H.PINKY_PIP, H.PINKY_DIP)
    dip = ang(H.PINKY_PIP, H.PINKY_DIP, H.PINKY_TIP)
    if pip > 160 and dip > 160:
        cnt += 1

    # Thumb (use MCP & IP angles + distance from index MCP)
    mcp_ang = ang(H.THUMB_IP, H.THUMB_MCP, H.THUMB_CMC)
    ip_ang  = ang(H.THUMB_TIP, H.THUMB_IP, H.THUMB_MCP)
    # distance thumb tip to index MCP (normalized by hand diag)
    dx = lm[H.THUMB_TIP].x - lm[H.INDEX_FINGER_MCP].x
    dy = lm[H.THUMB_TIP].y - lm[H.INDEX_FINGER_MCP].y
    tip_to_index = math.hypot(dx, dy) / diag

    if mcp_ang > 160 and ip_ang > 160 and tip_to_index > 0.13:
        cnt += 1

    return cnt

def get_alphabet(myHand):
    """
    Rule-based alphabet detector (A..Z subset). Returns '' if uncertain.
    myHand: list of [x, y] pixel landmarks (len 21).
    """
    try:
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

# =================== Camera & State ===================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    raise SystemExit

last_stable_letter = ['','']  # per-hand stable letter
buffers_letter = [collections.deque(maxlen=BUFFER_SIZE) for _ in range(MAX_HANDS)]
buffers_ok = [collections.deque(maxlen=BUFFER_SIZE) for _ in range(MAX_HANDS)]
ok_stable = [False for _ in range(MAX_HANDS)]

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

        # FPS (EMA)
        now = time.time()
        dt = now - fps_ts
        if dt > 0:
            fps = (0.9 * fps) + (0.1 * (1.0 / dt))
        fps_ts = now
        draw_label(img, f"FPS: {fps:.1f}", (10, 30), (0, 255, 0))

        total_fingers = 0

        if results.multi_hand_landmarks:
            # Pair each landmark set with handedness label
            handed_pairs = list(zip(results.multi_hand_landmarks, results.multi_handedness))
            handed_pairs = handed_pairs[:MAX_HANDS]

            # First pass: compute per-hand counts and accumulate total
            per_hand_counts = []
            for hand_landmarks, handness in handed_pairs:
                c = count_fingers(hand_landmarks)
                total_fingers += c
                label = handness.classification[0].label  # "Left" or "Right"
                per_hand_counts.append((label, c, hand_landmarks))

            # Draw total finger count (BLUE as requested)
            draw_label(img, f"Finger Count: {total_fingers}", (10, 60), (255, 0, 0))

            # Per-hand alphabet + OK detection and per-hand count labels
            for hand_idx, (label, c, hand_landmarks) in enumerate(per_hand_counts):
                # Pixel landmarks for alphabet rules
                h, w, _ = img.shape
                myHand = []
                for lm in hand_landmarks.landmark:
                    myHand.append([int(lm.x * w), int(lm.y * h)])

                # Draw skeleton
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Label positions near wrist
                base_x, base_y = myHand[0][0], myHand[0][1] if myHand else (10, 90)

                # Show per-hand finger count with handedness (cyan)
                draw_label(img, f"{'L' if label=='Left' else 'R'} = {c}", (base_x, min(h-10, base_y + 25)), (255, 255, 0))

                # 1) OK detection with smoothing
                raw_ok = is_ok_pinch(hand_landmarks)
                buffers_ok[hand_idx].append(1 if raw_ok else 0)
                stable_ok = sum(buffers_ok[hand_idx]) >= STABLE_THRESHOLD

                if SHOW_OK and stable_ok:
                    draw_label(img, "OK", (base_x, max(30, base_y - 10)), (0, 200, 0))
                    if SPEAK_OK and not ok_stable[hand_idx]:
                        if gate.can_speak("OK", now=time.time()):
                            tts.speak("OK")
                    ok_stable[hand_idx] = True
                    continue
                else:
                    ok_stable[hand_idx] = False

                # 2) Alphabet detection with smoothing
                raw_letter = get_alphabet(myHand)
                buffers_letter[hand_idx].append(raw_letter)
                stable_letter = most_common_nonempty(buffers_letter[hand_idx])

                # Speak only when a new stable letter appears (per hand)
                if SPEAK_LETTERS and stable_letter and stable_letter != last_stable_letter[hand_idx]:
                    if gate.can_speak(stable_letter, now=time.time()):
                        tts.speak(stable_letter)
                    last_stable_letter[hand_idx] = stable_letter

                # Draw letter label
                if stable_letter:
                    draw_label(img, f"{stable_letter}", (base_x, max(30, base_y - 10)))
                elif raw_letter:
                    draw_label(img, f"? {raw_letter}", (base_x, max(30, base_y - 10)), (255, 255, 0))

        else:
            # No hands
            draw_label(img, "Finger Count: 0", (10, 60), (255, 0, 0))
            draw_label(img, "No hand detected", (10, 90), (0, 0, 255))

        cv2.imshow("Unified: Alphabet (TTS) + Finger Count (L/R visual)", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    tts.shutdown()
