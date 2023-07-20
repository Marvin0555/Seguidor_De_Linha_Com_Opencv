import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
Starus_esteria = "OFF"

while True:

    _, frame = cap.read()
    height, width, dd = frame.shape
    belt = frame[0: 720, 500: 780 ]
    mask = cv2.cvtColor(belt, cv2.COLOR_BGR2HSV)
    cx = int(width / 2)
    cy = int(height / 2)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 3)
    cv2.line(frame, (cx, 0), (cx, height), (0, 0, 0), 3)

    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(mask, low_red, high_red)
    
    #erosão e dilatação
    erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))

    #erosão de rgb
    red_mask = cv2.erode(red_mask,erodeElement)

    #dilatação de rgb
    red_mask = cv2.erode(red_mask,dilateElement)

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in red_contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)
        if area > 100 and area < 300:
            cv2.rectangle(belt, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv2.putText(belt, str(area), (x, y), 1, 1, (0,255,0))
            moment = cv2.moments(cnt)
            area = moment['m00']
            qcentro_x = int(moment['m10']/area)
            qcentro_y = int(moment['m01']/area)
            cv2.circle(belt, (qcentro_x , qcentro_y), 10, (0,255,0), -1)

            # Coordenadas absolutas dos pontos extremos esquerdo e direito do retângulo em relação ao frame
            ponto_esquerdo_x = x 
            ponto_direito_x = x + w

            # Coordenadas absolutas da lina
            centro_x = 140

            # Calcula a distância entre os pontos extremos e o centro da linha
            distancia_esquerdo = abs(centro_x - ponto_esquerdo_x)
            distancia_direito = abs(centro_x - ponto_direito_x)

            cv2.putText(frame, f"Distância esquerda: {distancia_esquerdo}, Distância direita: {distancia_direito}", (0, 50), 2, 1, (0, 0, 255), 2)
        break
  
    cv2.imshow("Red", frame)
    cv2.imshow("parte", belt)

    key = cv2.waitKey(1)
    if key == 27:
        break