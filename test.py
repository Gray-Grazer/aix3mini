import face_recognition
import numpy as np
import os
from datetime import datetime

import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont


def createDirectory(directory, num_folder):
    for index in range(num_folder):
        folder_path = f"{directory}{index + 1}"
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                # 저장할 파일명
                unique_filename = generate_unique_filename(comparison_image)
                pil_image.save(os.path.join(folder_path, unique_filename))
                
        except OSError:
            print("Error: 폴더 경로 생성에 실패하였습니다.")


def generate_unique_filename(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    extension = os.path.splitext(file_path)[1][1:]  # [1:]를 사용하여 '.'을 제외한 확장자를 가져옴
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{base_name}_{timestamp}.{extension}"
    return unique_filename

# ============== 1-2. 인식시킬 다중 이미지 로드 및 페이스 인코딩 ================

load_error_text = ""
def load_and_encode_images(directory_path):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
         
            # 인식시킬 이미지 로드
            give_image = face_recognition.load_image_file(image_path)
            give_face_encodings = face_recognition.face_encodings(give_image)

            # 얼굴이 없는 이미지의 경우 건너뛰기
            if give_face_encodings:
                 # 첫 번째 얼굴 인코딩 가져오기
                give_encodings = give_face_encodings[0]
                name = os.path.splitext(filename)[0]

            # 페이스 인코딩 후 라벨링 작업
                known_face_encodings.append(give_encodings)
                known_face_names.append(name)

    return known_face_encodings, known_face_names

directory_path = "image"
known_face_encodings, known_face_names = load_and_encode_images(directory_path)

# ===========================================================================

# ============== 1-1. 인식시킬 단일 이미지 로드 및 페이스 인코딩 ================
# give_image = face_recognition.load_image_file("image\\unknown4.jpg")
# give_encodings = face_recognition.face_encodings(give_image)[0]

## 페이스 인코딩 후 라벨링 작업
# known_face_encodings = [
#     give_encodings
# ]
# known_face_names = [
#     "human1"
# ]
# ===========================================================================

# 2-1. GPU??
# gpuApp = FaceAnalysis(providers=['CPUExecutionProvider'])
# gpuImg = gpuApp.prepare(ctx_id=0, det_size=(640, 640))

# 비교할 이미지 읽기

## 이미지 파일 경로
comparison_image = "input\image\know1.jpg"
# unknown_image = cv2.imread("image\\unknown1.jpg")
unknown_image = face_recognition.load_image_file(comparison_image)

# ======================= unknown_image = cv2.imread =========================

## cv2.imread로 비교할 이미지를 읽었을 경우 모델 속도 개선을 위한 이미지 크기 조정
## small_frame = cv2.resize(unknown_image, (0, 0), fx=0.25, fy=0.25)
## rgb_small_frame = np.array(small_frame[:, :, ::-1])

## unknown 이미지에서 얼굴찾고 페이스 인코딩
# face_locations = face_recognition.face_locations(rgb_small_frame)
# face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
# ============================================================================

# ============= unknown_image = face_recognition.load_image_file =============

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)
# ============================================================================

# 유사한 이미지에 name을 라벨링하고 라벨링 시 폰트의 크기를 지정할 객체를 할당
face_names = []
font = ImageFont.load_default()

# ======================= unknown_image = cv2.imread =========================

# unknown 이미지와 로드한 이미지를 비교
# for face_encoding in face_encodings:
#     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#     name = "Unknown"

#     # 로드한 사진과 unknown 이미지의 유사도 측정 후 유사한 이미지에 name을 라벨링
#     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#     best_match_index = np.argmin(face_distances)
#     if matches[best_match_index]:
#         name = known_face_names[best_match_index]
#         face_names.append(name)
#     face_names.append(name)

# 비교를 위해 크기를 줄였던 unknown 이미지의 크기를 원래대로,
# for (top, right, bottom, left), name in zip(face_locations, face_names):
#     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#     top *= 4
#     right *= 4
#     bottom *= 4
#     left *= 4

#     # 인식된 얼굴 주변에 사각박스 그리기
#     cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2)

#     # 인식된 얼굴에 라벨링 그리기
#     cv2.rectangle(unknown_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#     font = cv2.FONT_HERSHEY_DUPLEX
#     cv2.putText(unknown_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
# ============================================================================

# ============= unknown_image = face_recognition.load_image_file =============

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # 로드한 이미지와 unknown 이미지의 유사도 측정 후 유사한 이미지에 name을 라벨링
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # 인식된 얼굴 주변에 사각박스 그리기 using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # 인식된 얼굴에 라벨링 그리기
    text_height = draw.textlength(name, font=font)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
# ============================================================================

# ======================= unknown_image = cv2.imread =========================

# 그리기까지의 작업을 마친 unknown 이미지를 출력
# cv2.imshow('result_unknown Image', unknown_image)
# cv2.waitKey(0)
# ============================================================================

# 저장할 폴더명
base_save_path = "output"

# 생성될 폴더 갯수 = 비교할 인풋 이미지 갯수
num_folder = len(comparison_image)

# 폴더 생성
createDirectory(base_save_path, num_folder)


# 그리기까지의 작업을 마친 unknown 이미지를 출력하고 저장
pil_image.show()
# 선별한 이미지를 기존 경로에서 삭제
# os.remove(image_path)
