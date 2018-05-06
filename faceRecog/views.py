from django.shortcuts import render, redirect
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from . import dataset_fetch as df
from . import cascade as casc
from PIL import Image

from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

from settings import BASE_DIR
from records.models import Records, Presensi
from records.forms import RecordsForm, PresensiForm
import datetime
from django.shortcuts import get_object_or_404

# membuat halaman utama
def index(request):
    return render(request, 'index.html')
def errorImg(request):
    return render(request, 'error.html')

def create_dataset(request):
    # print request.POST
    userId = request.POST['userId']
    print cv2.__version__
    # deteksi wajah dengan menggunakan cascade image classifier
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/ml/haarcascade_frontalface_default.xml')
    # capture image dari webcam dan proses dan mendeteksi wajah
    # mengambil id video capture dengan waktu 0 s.
    cam = cv2.VideoCapture(0)

    # identifikasi
    # id disimpan di dalam userId dan menyimpan id dengan wajah sehingga nantinya dapat mengidentifikasi wajah tersebut
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    id = userId
    # untuk penghitung penamaan dataset
    sampleNum = 0
    # capture wajah satu per satu dan deteksi wajah dan menampilkannya
    while(True):
        # Capture citra
        # cam.read akan mengembalikan variabel status dan citra berwarna yang diambil
        ret, img = cam.read()
        # citra yang dikembalikan adalah citra berwarna tetapi agar klasifikasi dapat bekerja dibutuhkan citra grayscale
        # untuk melakukan konversi
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # untuk menyimpan citra wajah
        # deteksi semua citra dalam frame, dan akan mengembalikan koordinat wajah
        # Ini akan mendeteksi semua citra dalam frame saat ini, dan ini akan mengembalikan koordinat wajah
        # Mengambil gambar dan beberapa parameter lain untuk hasil yang akurat
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        #Di atas variabel 'wajah' bisa ada beberapa wajah jadi kita harus mendapatkan setiap wajah dan menggambar persegi panjang di sekitarnya.
        for(x,y,w,h) in faces:
            # Whenever the program captures the face, we will write that is a folder
            # Setiap kali program capture wajah, kami akan menulis itu adalah folder
            # Before capturing the face, we need to tell the script whose face it is
            # Sebelum menangkap wajah, kita perlu memberi tahu naskah yang wajahnya itu
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum+1
            # Saving the image dataset, but only the face part, cropping the rest
            cv2.imwrite(BASE_DIR+'/ml/dataset/user.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(250)

        #Showing the image in another window
        #Creates a window with window name "Face" and with the image img
        cv2.imshow("Face",img)
        #Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        #To get out of the loop
        if(sampleNum>35):
            break
    #releasing the cam
    cam.release()
    # destroying all the windows
    cv2.destroyAllWindows()

    return redirect('/')

def trainer(request):
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.

        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''
    import os
    from PIL import Image

    #Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #Path of the samples
    path = BASE_DIR+'/ml/dataset'

    # To get all the images, we need corresponing id
    def getImagesWithID(path):
        # create a list for the path for all the images that is available in the folder
        # from the path(dataset folder) this is listing all the directories and it is fetching the directories from each and every pictures
        # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #concatinate the path with the image name
        #print imagePaths

        # Now, we loop all the images and store that userid and the face with different image list
        faces = []
        Ids = []
        for imagePath in imagePaths:
            # First we have to open the image then we have to convert it into numpy array
            faceImg = Image.open(imagePath).convert('L') #convert it to grayscale
            # converting the PIL image to numpy array
            # @params takes image and convertion format
            faceNp = np.array(faceImg, 'uint8')
            # Now we need to get the user id, which we can get from the name of the picture
            # for this we have to slit the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
            # Then we split the second part with . splitter
            # Initially in string format so hance have to convert into int format
            ID = int(os.path.split(imagePath)[-1].split('.')[1]) # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
            # Images
            faces.append(faceNp)
            # Label
            Ids.append(ID)
            #print ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(Ids), np.array(faces)

    # Fetching ids and faces
    ids, faces = getImagesWithID(path)

    #Training the recognizer
    # For that we need face samples and corresponding labels
    recognizer.train(faces, ids)

    # Save the recognizer state so that we can access it later
    # recognizer.save(BASE_DIR+'/ml/recognizer/trainingData.yml')
    recognizer.write(BASE_DIR + '/ml/recognizer/trainingData.yml')
    cv2.destroyAllWindows()

    return redirect('/')


def detect(request):
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/ml/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    # creating recognizer
    rec = cv2.face.LBPHFaceRecognizer_create();
    # loading the training data
    rec.read(BASE_DIR+'/ml/recognizer/trainingData.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)

            getId,conf = rec.predict(gray[y:y+h, x:x+w]) #This will predict the id of the face

            print conf;
            if conf<35:
                userId = getId
                cv2.putText(img, "Detected",(x,y+h), font, 2, (0,255,0),2)
            else:
                cv2.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)

            # Printing that number below the face
            # @Prams cam image, id, location,font style, color, stroke

        cv2.imshow("Face",img)
        if(cv2.waitKey(1) == ord('q')):
            break
        elif(userId != 0):
            # insert to presensi
            now = datetime.datetime.now()
            pegawai = Records.objects.get(npp=userId)
            presensiInstance = {
                'pegawai': pegawai.id,
                'tanggal': now.strftime("%Y-%m-%d"),
                'jam_masuk': now.strftime("%H:%M"),
                "kehadiran": 1,
            }

            todayDate = now.strftime("%Y-%m-%d")
            checkAbsen = Presensi.objects.filter(pegawai=pegawai.id, tanggal=todayDate)
            if checkAbsen.count() > 0:
                for absen in checkAbsen:
                    absen.jam_pulang = now.strftime("%H:%M")
                    absen.save(update_fields=['jam_pulang'])
            else:
                presensi = PresensiForm(presensiInstance)
                if presensi.is_valid():
                    pres = presensi.save()

            # close camera
            cv2.waitKey(1000)
            cam.release()
            cv2.destroyAllWindows()
            return redirect('/records/details/'+str(userId))

    cam.release()
    cv2.destroyAllWindows()
    return redirect('/')
