from __future__ import print_function
import numpy as np
import face_recognition
# dlib 설치가 안되서 구글링으로 https://github.com/Daiera/some_Resources/blob/master/dlib-19.17.0-cp37-cp37m-win_amd64.whl 검색후 파일 다운
# 다운 파일 파이참폴더로 옮기고 pip install dlib-19.17.0-cp37-cp37m-win_amd64.whl     # pip install face_recognition

import argparse
import cv2
import os
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import sklearn.neighbors._base
from sklearn.neighbors import KNeighborsClassifier

# argparse : Python Script 실행 시, 하나의 Script 동작을 여러가지 상황에 따라 다르게 동작하도록 할 때 쓰인다.
parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
# 인수명을 'arg'와 같이 일반적으로 지정하면 arg가 된다.
args = parser.parse_args()
#  dnn 모듈에서의 readNetFromCaffe 함수를 사용해, caffeframework의 네트워크 모델을 읽어들인다.
# (위에서 미리 정의한 caffemodel과 prototxt를 인자로 받아와서) 하나의 Net 객체를 반환하게 된다.
net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')

knn_clf = pickle.load(open('./models/fr_knn.pkl', 'rb'))


def adjust_gamma(image, gamma=1.0):  # 감마 조정
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def preprocess(img):  # 이미지 전처리
    ### analysis
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(1):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_img.mean() < 130:
            img = adjust_gamma(img, 1.5)  # 감마 조정
        else:
            break
    return img


# vc = cv2.VideoCapture('./data/TAEYANG_ONLY_LOOK_AT_ME_MV.mp4')
cap = cv2.VideoCapture('./data/with_M.mp4')  # 사용할 동영상 넣으면 결과로 이 영상에서 모자이크처리와 라벨링 보여줌
# 비디오 캡쳐. 동영상을 cap에 저장. vc를 cap으로 바꿈


# 동영상 저장하는 코드 [추가]
width = int(cap.get(3))  # 가로 길이 가져오기
height = int(cap.get(4))  # 세로 길이 가져오기
fps = 20  # 초당 프레임
fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # VideoWriter_fourcc('M', 'J', 'P', 'G'): 모션 JPEG 코덱/ 코덱은 지정하는 함수
out = cv2.VideoWriter('output_label.avi', fcc, fps, (width, height))  # '저장할 파일명'
out_2 = cv2.VideoWriter('output_blur.avi', fcc, fps, (width, height))

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('length :', length)

# 라벨링 영상 불러오는 if문 namedWindow('show', 0)
# if args.with_draw == 'True':
#     cv2.namedWindow('show', 0)

for idx in range(length):
    img_bgr = cap.read()[1]
    if img_bgr is None:
        break
    # if idx%3 != 0: continue
    # if idx < 200: continue

    start = cv2.getTickCount()  # getTickCount : OS부팅할 때부터 지나간 시간을 msec 단위로 돌려주는 함수

    ### preprocess
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_bgr_ori = img_bgr.copy()
    img_bgr = preprocess(img_bgr)

    ### detection
    border = (img_bgr.shape[1] - img_bgr.shape[0]) // 2
    # copyMakeBorder : 이미지를 액자 형태로 만들 때 사용. 가장자리가 추가되는데, 이 함수의 6번째 인자에 따라 추가되는 가장자리의 형태가 결정
    img_bgr = cv2.copyMakeBorder(img_bgr,
                                 border,  # top
                                 border,  # bottom
                                 0,  # left
                                 0,  # right
                                 cv2.BORDER_CONSTANT,  # 일정한 색상의 테두리를 추가
                                 value=(0, 0, 0))

    (h, w) = img_bgr.shape[:2]
    # blob= 이미지, 사운드, 비디오와 같은 멀티미디어 데이터를 다룰 때 사용하는 라이브러리
    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # 영상으로부터 블롭을 생성합니다. 이 함수는 입력 image로부터 4차원 블롭 객체를 생성하여 반환
    net.setInput(blob)
    detections = net.forward()

    ### bbox
    list_bboxes = []
    list_confidence = []
    # list_dlib_rect = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
            continue
        # astype메서드 : 열의 요소의 dtype을 변경하는함수
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (l, t, r, b) = box.astype("int")  # l t r b

        original_vertical_length = b - t
        t = int(t + (original_vertical_length) * 0.15) - border
        b = int(b - (original_vertical_length) * 0.05) - border

        margin = ((b - t) - (r - l)) // 2  # 마진 : 여백
        l = l - margin if (b - t - r + l) % 2 == 0 else l - margin - 1
        r = r + margin
        refined_box = [t, r, b, l]
        list_bboxes.append(refined_box)
        list_confidence.append(confidence)

    ### facenet
    # 얼굴을 인코딩해준다. 부호화(encoding) 해주는것
    face_encodings = face_recognition.face_encodings(img_rgb, list_bboxes)
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    is_recognized = [closest_distances[0][i][0] <= 0.4 for i in range(len(list_bboxes))]
    list_reconized_face = [(pred, loc, conf) if rec else ("unknown", loc, conf) for pred, loc, rec, conf in
                           zip(knn_clf.predict(face_encodings), list_bboxes, is_recognized, list_confidence)]
    # print (list_reconized_face)        #list_reconized_face : 리스트화된 인식된 얼굴

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print('%d, elapsed time: %.3fms' % (idx, time))

    ### blurring
    img_bgr_blur = img_bgr_ori.copy()
    for name, bbox, conf in list_reconized_face:
        t, r, b, l = bbox
        if name == 'unknown':
            face = img_bgr_blur[t:b, l:r]
            small = cv2.resize(face, None, fx=.05, fy=.05, interpolation=cv2.INTER_NEAREST)
            blurred_face = cv2.resize(small, (face.shape[:2]), interpolation=cv2.INTER_NEAREST)
            img_bgr_blur[t:b, l:r] = blurred_face

    ### draw rectangle bbox
    if args.with_draw == 'True':
        # Image.fromarray()를사용하여, numpy 배열을 PIL Image로 변환하고, 이를 save()를 통해 저장
        source_img = Image.fromarray(img_bgr_ori)
        # 즉, NumPy 배열을 PIL 이미지로 변환하는 코드
        draw = ImageDraw.Draw(source_img)
        for name, bbox, confidence in list_reconized_face:
            t, r, b, l = bbox
            # print (int((r-l)/img_bgr_ori.shape[1]*100))
            font_size = int((r - l) / img_bgr_ori.shape[1] * 100)

            draw.rectangle(((l, t), (r, b)), outline=(0, 255, 128))

            draw.rectangle(((l, t - font_size - 2), (r, t + 2)), fill=(0, 255, 128))
            draw.text((l, t - font_size), name, font=ImageFont.truetype('./BMDOHYEON_TTF.TTF', font_size),
                      fill=(0, 0, 0, 0))
        # np.asarray : 입력된 데이터를 numpy.ndarray() 형식으로 만들어줍니다.
        # ndarray는 Numpy의 핵심인 '다차원 행렬 자료구조 클래스'
        show = np.asarray(source_img)
        # 라벨링 영상 불러오고 저장하는 것
        cv2.imshow('show', show)  # 라벨링 영상 보여주는 것
        out.write(show)  # 라벨링 한 영상 avi로 저장
        # 모자이크 영상 불러오고 저장하는 것
        cv2.imshow('blur', img_bgr_blur)  # 모자이크 한 영상 보여주는 것(읽어오기)
        out_2.write(img_bgr_blur)  # 모자이크 처리한 영상 avi로 저장
        key = cv2.waitKey(30)
        if key == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()

# 동영상 파일 저장하는 원본 코드(복붙한 것)
# cap = cv2.VideoCapture('./data/with_M.mp4')     # 사용할 동영상 넣으면 결과로 이 영상에서 모자이크처리와 라벨링 보여줌
# 비디오 캡쳐. 동영상을 cap에 저장.
# width = int(cap.get(3)) # 가로 길이 가져오기
# height = int(cap.get(4)) # 세로 길이 가져오기
# fps = 20
#
# fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# out = cv2.VideoWriter('output_blur.avi', fcc, fps, (width, height))
# print(fps)
# print(fcc)
# print(width)
# print(height)
# while (cap.isOpened()) :
#
#     ret, frame = cap.read()
#
#     if ret :
#         frame = cv2.flip(frame, 0)
#         out.write(frame)
#         cv2.imshow('frame', frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q') : break
#
#     else :
#         print("Fail to read frame!")
#         break


### opencv text, box drawing
# cv2.rectangle(img_bgr_blur, (l, t), (r, b), (0, 255, 0), 2)

# cv2.rectangle(img_bgr_ori, (l, t), (r, b), (0, 255, 128), 2)
# text = "%s: %.2f" % (name,confidence)
# text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
# y = t #- 1 if t - 1 > 1 else t + 1
# cv2.rectangle(img_bgr_ori,
#             (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
# cv2.putText(img_bgr_ori, text, (l, y),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)