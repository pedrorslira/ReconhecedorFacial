import cv2 as cv
import os

cascPath = "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml" #aqui é inserido o path para o reconhecedor de faces
faceCascade = cv.CascadeClassifier(cascPath)
fonte = cv.FONT_HERSHEY_SIMPLEX
video = cv.VideoCapture(0)
num_img = 0 #contador de imagens de cada rosto
total_img = 50 #total de imagens de cada rosto que vão ser salvas
#tamanho das imagens que vão ser salvas no dataset:
largura = 130
altura = 100

nomeNovaPasta = input('Informe o nome da pessoa que vai ser analisada por reconhecimento facial\n')

#definindo a pasta do dataset de rostos
path = os.path.join('dataset', nomeNovaPasta) 
if not os.path.isdir(path): 
   os.mkdir('./'+path) 
   
while True:   
    roi_gray = None
    ret, frame = video.read()  
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
            face_resize = cv.resize(roi_gray, (largura, altura)) 
            cv.putText(frame,'Rosto',(x, y-20), fonte, 2,(1, 150, 250),5)
            if num_img < total_img:
                num_img += 1
                print('Analisando... Imagem '+ str(num_img) + ' de ' + str(total_img))
                cv.imwrite('%s/%s.png' % (path, num_img), face_resize)
            elif num_img == total_img:
                num_img += 1
                print('Análise Completa!')   
    cv.imshow('Webcam', frame)
    tecla = cv.waitKey(1)
    if tecla == 27: #A tecla Esc fecha o programa
        break

video.release()
cv.destroyAllWindows()