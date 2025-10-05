import cv2
import mediapipe as mp
import pyttsx3  # Thêm thư viện pyttsx3 để chuyển văn bản thành giọng nói

# Khởi tạo MediaPipe cho bàn tay
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo AI giọng nói
engine = pyttsx3.init()

# Cài đặt AI giọng nói
voices = engine.getProperty('voices')
for voice in voices:
    engine.setProperty('voice', voice.id)
    break

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Khởi tạo Hands từ MediaPipe
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

previous_finger_count = -1  # Biến lưu số ngón tay đếm được trước đó

while True:
    # Đọc một frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Không thể đọc khung hình.")
        break

    # Lật ngược khung hình để hiển thị ảnh theo hướng người dùng nhìn
    frame = cv2.flip(frame, 1)
    
    # Chuyển đổi hình ảnh sang màu RGB cho MediaPipe xử lý
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Nếu phát hiện bàn tay
    if results.multi_hand_landmarks:
        total_finger_count = 0  # Tổng số ngón tay đang giơ từ cả hai tay

        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ các điểm mốc bàn tay lên khung hình
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Đếm số ngón tay đang giơ cho từng tay
            finger_count = 0

            # Kiểm tra ngón cái (index 4) - Ngón cái phải lên trên ngón cái gốc (index 2)
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
                finger_count += 1

            # Kiểm tra các ngón tay còn lại (ngón trỏ, giữa, đeo, út)
            # Đếm các ngón tay khi đầu ngón tay (TIP) ở trên vị trí khớp (PIP)
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                finger_count += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                finger_count += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
                finger_count += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
                finger_count += 1

            # Cộng số ngón tay từ mỗi tay vào tổng số ngón tay
            total_finger_count += finger_count

            # Kiểm tra cử chỉ Unlike: Ngón cái và ngón trỏ chạm vào nhau
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = abs(thumb_tip.x - index_finger_tip.x) + abs(thumb_tip.y - index_finger_tip.y)
            if distance < 0.05:  # Nếu khoảng cách giữa ngón cái và ngón trỏ nhỏ hơn 0.05 thì là cử chỉ "unlike"
                cv2.putText(frame, "Oke", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                engine.say("Oke")  # Đọc "Unlike"
                engine.runAndWait()

        # Hiển thị tổng số ngón tay đang giơ
        cv2.putText(frame, f"So ngon tay: {total_finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Chỉ đọc giọng nói khi số ngón tay thay đổi
        if total_finger_count != previous_finger_count:
            engine.say(f"{total_finger_count}")  # Đọc giọng nói tiếng Việt
            engine.runAndWait()

        # Lưu lại số ngón tay đã đếm
        previous_finger_count = total_finger_count
    else:
        # Nếu không phát hiện bàn tay, chỉ hiển thị "Finger Count: Unknown" mà không có giọng nói
        cv2.putText(frame, "Khong nhan dien duoc cu chi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Hiển thị hình ảnh lên màn hình
    cv2.imshow("So ngon tay", frame)

    # Nhấn 'Esc' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
