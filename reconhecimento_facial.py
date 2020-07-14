import cv2 as cv
import os
import numpy as np

cascPath = "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml" #aqui é inserido o path para o reconhecedor de faces
faceCascade = cv.CascadeClassifier(cascPath)
fonte = cv.FONT_HERSHEY_SIMPLEX
fonte2 = cv.FONT_HERSHEY_PLAIN
video_capture = cv.VideoCapture(0)
#tamanho das imagens que estão no dataset:
largura = 130
altura = 100

imagens = []
lables = []
nomes = {}
id = 0

for (subdiretorios, diretorios, arquivos) in os.walk('dataset'): 
    for subdiretorios in diretorios: 
        nomes[id] = subdiretorios 
        path_arquivo = os.path.join('dataset', subdiretorios) 
        for nome_arquivo in os.listdir(path_arquivo): 
            path = path_arquivo + '/' + nome_arquivo
            lable = id
            imagens.append(cv.imread(path, 0)) 
            lables.append(int(lable)) 
        id += 1

(imagens, lables) = [np.array(lis) for lis in [imagens, lables]]
  
#treinando o reconhecedor de face:
reconhecedor = cv.face.LBPHFaceRecognizer_create() 
reconhecedor.train(imagens, lables) 
face_cascade = cv.CascadeClassifier(cascPath) 

print('Reconhecendo Rosto...') 
  
while True:
    roi_gray = None
    ret, frame = video_capture.read()  
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostos = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in rostos:
            cv.rectangle(frame, (x, y), (x+w, y+h), (1, 150, 250), 3)
            roi_gray = gray[y:y+h, x:x+w]
            rosto = cv.resize(roi_gray, (largura, altura)) 
            #Reconhecer o rosto que está na frente da câmera:
            prediction = reconhecedor.predict(rosto) 
            if prediction[1]<150: 
                cv.putText(frame, '%s - %.0f' % (nomes[prediction[0]], prediction[1]), (x, y-20), fonte2, 2, (0, 255, 0),2) 
            else: 
                cv.putText(frame, 'Desconhecido', (x, y-20), fonte2, 2, (0, 255, 0),2)                    
    cv.imshow('Webcam', frame)
    tecla = cv.waitKey(1)
    if tecla == 27: #A tecla Esc fecha o programa
        break
video_capture.release()
cv.destroyAllWindows()