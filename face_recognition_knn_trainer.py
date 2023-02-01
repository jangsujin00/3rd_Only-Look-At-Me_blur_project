# -*- coding: utf-8 -*-
"""
This is an example of using the k-nearest-neighbors(knn) algorithm for face recognition.

When should I use this example?
This example is useful when you whish to recognize a large set of known people,
and make a prediction for an unkown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled(known) faces, and can then predict the person
in an unkown image by finding the k most similar faces(images with closet face-features under eucledian distance) in its training set,
and performing a majority vote(possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden and two images of Obama, 
The result would be 'Obama'.
*This implemententation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:
-First, prepare a set of images of the known people you want to recognize.
 Organize the images in a single directory with a sub-directory for each known person.
-Then, call the 'train' function with the appropriate parameters.
 make sure to pass in the 'model_save_path' if you want to re-use the model without having to re-train it. 
-After training the model, you can call 'predict' to recognize the person in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:
$ pip3 install scikit-learn
"""

import os
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle       #  pickle 모듈 사용. 파이썬에서 만들어지는 것은 뭐든지 다 파일에 적을 수 있음
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from glob import glob
# from face_recognition.cli import image_files_in_folder
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path="", n_neighbors=None, knn_algo='ball_tree', verbose=False):        # Ball-Tree는 범위를 기준으로 차원을 내림차순 정렬한 후, 서브트리에 속한 범위의 반을 기준으로 KD-Tree를 적용
        # 디렉토리, disk 모델 저장 경로, 분류에 가중치를 부여할 이웃 수, 기본 데이터 구조, 진행상황 보여줌 - x )                                                               # knn : K-Nearest Neighbors ('K개의 근접 이웃')
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.     # 이름과 함께 알려진 각 사람에 대한 하위 디렉토리를 포함하는 디렉토리.

     (View in source code to see train_dir example tree structure)
     (train_dir 예제 트리 구조를 보려면 소스 코드에서 보기)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model of disk           # disk 모델 저장 경로
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified.   # 분류에 가중치를 부여할 이웃 수. 지정하지 않으면 자동으로 선택
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree           # knn.default를 지원하는 기본 데이터 구조는 ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.             # 주어진 데이터가 학습되는 knn 분류기 반환
    """
    X = []
    y = []
    for class_dir in listdir(train_dir):            # dir( ) [directory] : 해당 객체가 어떤 변수와 메소드(method)를 가지고 있는지 나열
        if not isdir(join(train_dir, class_dir)):   # isdir() : 디렉토리 존재 여부 확인 - TRUE/False로 반환      # join() : 리스트를 문자열로 합쳐줌
            continue
        for img_path in sorted(glob(join(train_dir, class_dir, '*'))):          # glob() : 인자로 받은 패턴과 이름이 일치하는 모든 파일과 디렉터리의 리스트를 반환 (패턴을 그냥 *라고 주면 모든 파일과 디렉터리)
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:                 # 학습 진행 상황 보여주는
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(
                        faces_bboxes) < 1 else "found more than one face"))
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))      # round:반올림(sqrt:제곱근( X의 길이 ))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')       # clf (Classifier) 분류기
    knn_clf.fit(X, y)

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:      # model_save_path 파일을 새로 열어서 f라는 이름을 붙임. wb는 바이트(byte) 형식으로 쓰겠다(write)의 의미
            pickle.dump(knn_clf, f)                 # dump는 knn_clf를 파일 f에 기록. (파이썬이 실행되는 경로에 knn_clf파일이 만들어짐)
    return knn_clf


def predict(X_img_path, knn_clf=None, model_save_path="", DIST_THRESH=.4):      # 거리 한계점 (distance_threshold= .4) - 숫자 클수록 학습된 사람으로 잘못 분류할 가능성 큼
    """    # 인식된 이미지 경로, knn 분류자 객체, knn 분류자의 경로, 거리 한계점
    recognizes faces in given image, based on a trained knn classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.    # knn 분류자 객체. 지정하지 않으면 model_save_path를 지정해야 함
    :param model_save_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.  # pickled knn 분류자의 경로. 지정되지 않은 경우 model_save_path는 knn_clf여야 함.
    :param DIST_THRESH: (optional) distance threshold in knn classification. the larger it is, the more chance of misclassifying an unknown person to a known one.  # 크기가 클수록 알려지지 않은 사람을 알려진 사람으로 잘못 분류할 가능성이 커짐.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...]. # 이미지에서 인식된 얼굴의 이름 및 위치 리스트 반환
        For faces of unrecognized persons, the name 'N/A' will be passed.
    """

    if (not isfile(X_img_path)) or (splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS):   # not 연산자가 or 연산자보다 먼저
        raise Exception("invalid image path: {}".format(X_img_path))                          # raise Exception : 오류 발생시 구문 출력

    if knn_clf is None and model_save_path == "":
        raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")     # raise Exception : 오류 발생시 구문 출력
                        # knn_clf나 model_save_path를 통해 knn 분류자를 제공해야 함.
    if knn_clf is None:
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_img = face_recognition.load_image_file(X_img_path)  # 3개의 RGB
    print(X_img.shape)      #(가로, 세로, 3)
    # show = cv2.cvtColor(X_img, cv2.COLOR_RGB2BGR)
    # cv2.imshow('show', show)
    # cv2.waitKey()
    X_faces_loc = face_locations(X_img)  # (top, right, bottom, left)
    if len(X_faces_loc) == 0:
        return []
    print(X_faces_loc[0])           # (top, right, bottom, left) : 얼굴 좌표
    start = cv2.getTickCount()              # GetTickCount() 함수는 마이크로 초마다 시간값 받아옴.
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print('prediction time: %.2fms' % time)
    is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    # 신뢰도가 높지 않은 클래스 및 선별 분류를 예측
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]


def draw_preds(img_path, preds):
    """     # 인식된 이미지 경로, 예측 결과
    shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param preds: results of the predict function
    :return:
    """
    source_img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for pred in preds:               # 반복문으로 리스트의 모든 내용 출력    # ex) item_list = [1, 2, 3, 4, 5]
                                                                        #     for item in item_list:
                                                                        #        print(item)
        loc = pred[1]
        name = pred[0]
        # (top, right, bottom, left) => (left,top,right,bottom)
        draw.rectangle(((loc[3], loc[0]), (loc[1], loc[2])), outline="red")
        # draw.text((loc[3], loc[0] - 30), name, font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30))
        draw.text((loc[3], loc[0] - 21), name, font=ImageFont.truetype('./BMDOHYEON_TTF.TTF', 20))
    source_img.show()


if __name__ == "__main__":
    knn_clf = train("./data/train", model_save_path='./models/fr_knn.pkl')  # train폴더에 학습시킬 이미지 폴더로 저장해야 나중에 라벨링 처리됨
    for img_path in listdir("./data/test"):
        # preds = predict(join("./data/test", img_path), knn_clf=knn_clf)
        preds = predict(join("./data/test", img_path), model_save_path='./models/fr_knn.pkl', DIST_THRESH=0.4)      # 거리 한계점 (distance_threshold= 0.4)
                        #       인식된 이미지 경로      ,             knn 분류자의 경로         ,    거리 한계점
        print(os.path.basename(img_path), preds)
        # draw_preds(join("./data/test", img_path), preds)

# predict(X_img_path, knn_clf=None, model_save_path="", DIST_THRESH=.4):
        # 인식된 이미지 경로, knn 분류자 객체, knn 분류자의 경로, 거리 한계점
