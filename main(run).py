import glob
import os
import face_recognition
import cv2
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog

from capture import Capture
from test1 import imagesData, allA, a, b

# current_run = 'train'
current_run = 'run'
input_image = 'filePicker'
# input_image = 'camShot'
mode_selected = 0   # select which mode to start run with..
# mode_selected = 1
tolerance=0.85  #un-accuracy "error" percentage
encoding_images_path = 'encoded images files'
images_path_F = 'photos/womens/'        # Train From path
images_path = ''
Female_list = []


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 20

# image = 'photos/dataset_brand_cut/maybeline/fdf (2) before.png'
image = 'photos/test/img.png'






class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings_f = []
        self.known_face_names_f = []
        self.frame_resizing = 0.25


    def load_encoding_images(self, images_path, category):
        print(f'{category} Images Encoding:')
        if current_run == 'train':
            for img_path in images_path:
                img = cv2.imread(img_path)  # read and show image
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to another color space, There are more than 150 color-space
                basename = os.path.basename(img_path)
                (filename, ext) = os.path.splitext(basename)  # split colored/multi-channel image into separate single-channel images
                # filename = str(os.path.dirname(img_path).split('/')[-1].split('\\')[-1])
                filename = img_path
                try:
                    img_encoding = face_recognition.face_encodings(rgb_img)[0]  #return the 128-dimension face encoding for each face
                except:
                    1
                self.known_face_encodings_f.append(img_encoding)
                self.known_face_names_f.append(filename)
            with open(f'{encoding_images_path}/{category}_encoding.txt', "wb") as fp:  # Pickling
                pickle.dump(self.known_face_encodings_f, fp)

            with open(f'{encoding_images_path}/{category}_names.txt', "wb") as fp:  # Pickling
                pickle.dump(self.known_face_names_f, fp)
            self.known_face_encodings_f = []
            self.known_face_names_f = []
            with open(f'{encoding_images_path}/{category}_encoding.txt', "rb") as fp:  # Unpickling
                self.known_face_encodings_f = pickle.load(fp)
            with open(f'{encoding_images_path}/{category}_names.txt', "rb") as fp:  # Unpickling
                self.known_face_names_f = pickle.load(fp)
            print(f"{category} Encoding images loaded")

        else:
            with open(f'{encoding_images_path}/{category}_encoding.txt', "rb") as fp:  # Unpickling
                self.known_face_encodings_f = pickle.load(fp)
            with open(f'{encoding_images_path}/{category}_names.txt', "rb") as fp:  # Unpickling
                self.known_face_names_f = pickle.load(fp)
            print(f"{category} Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)  # to change photo size
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # convert image to another color space, There are more than 150 color-space
        face_locations = face_recognition.face_locations(rgb_small_frame)  # bounding boxes of human faces
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # return the 128-dimension face encoding for each face
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings_f, face_encoding, tolerance=tolerance)  # Compare faces to see if they match
            name = "Can`t Detect"
            face_distances = face_recognition.face_distance(self.known_face_encodings_f, face_encoding)  # get distance (un-similarity) for each comparison face
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names_f[best_match_index]
            else:
                print('unknown detected!!')
            if len(face_names) < 2:
                face_names.append(name)
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names#, url


def main_GUI():
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    Bg = cv2.imread('photos/img.png', cv2.IMREAD_UNCHANGED)
    Bg = cv2.resize(Bg, (1000, 700), interpolation=cv2.INTER_AREA)
    # resized = cv2.resize(img, (540, 380), interpolation=cv2.INTER_AREA)
    # frame = resized
    # imgBG[170:650, 725:1365] = frame
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # creating a smaller frame for better optimization:
    face_locations, face_names = sfr.detect_known_faces(img)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        # try:
        cv2.putText(img, str(os.path.dirname(name).split('/')[-1].split('\\')[-1]), (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)  # to add text into an image
        cv2.putText(Bg, "The best cosmetic product for you is:", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (200, 0, 0), 2)  # to add text into an image
        cv2.putText(Bg, "Before:", (600, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)  # to add text into an image
        cv2.putText(Bg, "And here is a picture of the", (10, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)  # to add text into an image
        cv2.putText(Bg, "person who looks like you the", (10, 480), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)  # to add text into an image
        cv2.putText(Bg, "most use this product:", (10, 510), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)  # to add text into an image
        cv2.putText(Bg, "After:", (800, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)  # to add text into an image
        name = name.split('/')[0]+'\\'+name.split('/')[-1]
        collectedPath = ''
        for i in range(len(name.split('\\'))-1):
            collectedPath += name.split('\\')[i]+"\\"
        aft_ImgName = cv2.imread(name.replace('\\','/'), cv2.IMREAD_UNCHANGED)
        bef_ImgName = cv2.imread(name.replace('after', 'before').replace('\\','/'), cv2.IMREAD_UNCHANGED)
        aft_ImgName = cv2.resize(aft_ImgName, (150, 300), interpolation=cv2.INTER_AREA)
        bef_ImgName = cv2.resize(bef_ImgName, (150, 300), interpolation=cv2.INTER_AREA)
        # print(lastImgName)
        brand = collectedPath[:]+'brand.png'
        brand = brand.replace('\\','/')
        # print('collectedPath: '+brand)
        brImg = cv2.imread(brand, cv2.IMREAD_UNCHANGED)
        # except:
        #     1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 3)  # draw rectangle on photo “usually used for face boundaries”

        h, w, o = img.shape
        if h < 250 or w < 250:
            resized = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
        elif h < 400 or w < 400:
            frame = img
        else:
            resized = cv2.resize(img, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_AREA)

    try: Bg[15:240, 650:875] = brImg
    except: Bg[15:240, 650:875, 2] = brImg
    try: Bg[300:300+int(h/2), 210:210+int(w/2)] = resized[:,:,:3]
    except:
        try: Bg[300:300+int(h*2), 210:210+int(w*2)] = resized[:,:,:3]
        except: Bg[300:300+int(h), 210:210+int(w)] = resized[:,:,:3]
    Bg[370:670, 600:750] = bef_ImgName
    Bg[370:670, 800:950] = aft_ImgName
    # try: Bg[0:225, 600:825, 2] = brImg
    # except: Bg[10:235, 550:775] = brImg
    # try: Bg[250:650, 150:350] = bef_ImgName
    # except: Bg[250:650, 150:300] = bef_ImgName
    # try: Bg[250:650, 650:850] = aft_ImgName
    # except: frame[250:550, 650:800] = aft_ImgName
    cv2.waitKey(1)  # read and show image
    cv2.imshow("3omar.hs Detection..", Bg)  # read and show image
def main_GUI1():
    inputImg = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img = cv2.imread('photos/img.png', cv2.IMREAD_UNCHANGED)
    brand = cv2.imread('photos/brands/loreal.png', cv2.IMREAD_UNCHANGED)
    frame = cv2.resize(img, (1000, 700), interpolation=cv2.INTER_AREA)
    # brand = cv2.resize(brand, (1000, 700), interpolation=cv2.INTER_AREA)
    img1 = cv2.imread('photos/dataset_brand_cut/mac/mac (1) after.png',cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('photos/dataset_brand_cut/mac/mac (1) before.png',cv2.IMREAD_UNCHANGED)
    h, w, o = img1.shape
    # img1 = cv2.resize(img1, (w*1.25, h*1.25), interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img1, (200, 400), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (200, 400), interpolation=cv2.INTER_AREA)
    # frame = resized
    # h,w,o = img.shape
    # if h<250 or w<250:
    # frame = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_AREA)
    # else:
    #     frame = img
    # imgBG[170:650, 725:1365] = frame
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # creating a smaller frame for better optimization:
    face_locations, face_names = sfr.detect_known_faces(inputImg)
    for face_loc, name in zip(face_locations, face_names):
        print('*'*20+name)
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        # try:
        #     cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)  # to add text into an image
        #     # print(name)
        # except:
        #     1
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 3)  # draw rectangle on photo “usually used for face boundaries”
        # frame = overlayPNG(img, img1)
    # frame[100:200, 50:400] = img1
    # frame[50:200, 100:400] = img1
    frame[0:225, 600:825, 2] = brand
    frame[250:650, 150:350] = img1
    frame[250:650, 650:850] = img2
    cv2.waitKey(1)  # read and show image
    cv2.imshow("3omar.hs Detection..", frame)  # read and show image

if input_image == 'filePicker':
    tk.Tk().withdraw()
    image = filedialog.askopenfilename()
elif input_image == 'camShot':
    image = Capture.Image()
imagesData()
sfr = SimpleFacerec()
# sfr.load_encoding_images(images_path_F, 'Female')
for i in range(len(list(b))):
    sfr.load_encoding_images(allA, list(b)[i])
# try:
while True:
    main_GUI()
# except: 1