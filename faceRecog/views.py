from django.shortcuts import render, redirect
import cv2
import numpy as np # numerical processing dengan python
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
from records.models import Records, Presensi, Citra
from records.forms import RecordsForm, PresensiForm, CitraForm
import datetime

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
            # Setiap kali program capture wajah, wajah tersebut disimpan di dalam folder
            # wajah di capture dan identifikasi melalui id
            sampleNum = sampleNum+1
            # Menyimpan image dataset, tetapi hanya bagian wajah
            cv2.imwrite(BASE_DIR+'/ml/dataset/user.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
            # titik awal dari persegi panjang akan menjadi x, y dan titik akhir akan menjadi x + lebar dan y + tinggi, warna dari citra gray
            citraName = 'user.'+str(id)+'.'+str(sampleNum)+'.jpg'

            # save image name on database
            pegawai = Records.objects.get(npp=str(id))
            citraInstance = {
                'npp': str(id),
                'citra_name': citraName,
                'pegawai': pegawai.id
            }
            checkCitra = Citra.objects.filter(npp=str(id), citra_name=citraName)
            if checkCitra.count() > 0:
                for citraUpdate in checkCitra:
                    citraUpdate.citra_name = citraName
                    citraUpdate.save(update_fields=['citra_name'])
            else:
                citra = CitraForm(citraInstance)
                if citra.is_valid():
                    cit = citra.save()

            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
            # sebelum melanjutkan ke perulangan selanjtunya, akan diberi waktu berhenti 250 ms
            cv2.waitKey(250)

        #menampilkan citra di window yang berbeda
        #menciptakan window dengan nama "Face" dan dengan citra img
        cv2.imshow("Face",img)
        #Sebelum menutupnya, kita perlu memberikan perintah tunggu selama 1 ms, jika tidak opencv tidak akan berfungsi
        cv2.waitKey(1)
        #untuk keluar dari perulangan
        if(sampleNum>35):
            break
    #menutup kamera
    cam.release()
    # menutup semua window
    cv2.destroyAllWindows()

    return redirect('/')

def trainer(request):

    import os
    from PIL import Image

    #membuat recognizer LBPH dengan fungsi yang sudah disediakan
    #oleh opencv dan memasukkan ke variabel
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #Path of the samples
    path = BASE_DIR+'/ml/dataset'

    # id yang sesuai diperlukan untuk mendapatkan semua citra
    def getImagesWithID(path):
        # membuat list untuk folder dataset untuk semua image yang tersedia di folder
        # dari folder dataset diambil direktori dari masing-masing gambar
        # menempatkan setiap gambar di 'f'
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        #print imagePaths

        # dilakukan looping pada semua gambar dan menyimpan userid dan wajah dengan daftar gambar yang berbeda
        faces = []
        Ids = []
        for imagePath in imagePaths:
            # pertama membuka gambar maka sebelumnya gambar diubah menjadi numpy array
            faceImg = Image.open(imagePath).convert('L') #konversikan menjadi grayscale
            # konversikan PIL image menjadi numpy array
            # ambil gambar dan mengkonversi format
            faceNp = np.array(faceImg, 'uint8')

            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            # Label
            Ids.append(ID)
            #cetak ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(Ids), np.array(faces)

    # mengambil id and wajah
    ids, faces = getImagesWithID(path)

    #melatih pengenalan wajah
    # untuk itu diperlukan sampel wajah dan id yang sesuai
    recognizer.train(faces, ids)

    #menulis recognizer yang sudah di training oleh program trainer
    recognizer.write(BASE_DIR + '/ml/recognizer/trainingData.yml')
    cv2.destroyAllWindows()

    return redirect('/')


def detect(request):
    # melakukan load/memuat classifier
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
        faces = faceDetect.detectMultiScale(gray, 1.3, 5) #deteksi ada tidaknya wajah pada citra, wajah yang ditemukan memiliki return value berupa tuple (x,y,width,height)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2) #membuat objek berbentuk persegi pada wajah jika terdeteksi

            getId,conf = rec.predict(gray[y:y+h, x:x+w])

            print conf;
            if conf<36:
                userId = getId
                cv2.putText(img, "Detected",(x,y+h), font, 2, (0,255,0),2)
            else:
                cv2.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)

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
