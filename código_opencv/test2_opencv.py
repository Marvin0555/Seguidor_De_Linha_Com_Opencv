import cv2
import numpy as np
import time

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
Starus_esteria = "OFF"

while True:

    _, frame = cap.read()
    height, width, dd = frame.shape
    belt = frame[0: 720, 500: 780 ]
    mask = cv2.cvtColor(belt, cv2.COLOR_BGR2HSV)
    cx = int(width // 2)
    cy = int(height // 2)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Extraia a região de interesse
    cv2.circle(frame, (cx , cy), 10, (0,255,0), -1)
    cv2.line(frame, (cx+100, 0), (cx+100, height), (100, 200, 50), 3)
    cv2.line(frame, (cx-100, 0), (cx-100, height), (100, 200, 50), 3)
    cv2.line(frame, (cx-100, cy), (cx+100, cy), (100, 30, 170), 3)

    # Red color
    low_red = np.array([99, 108, 131]) #np.array([161, 155, 84])
    high_red = np.array([180, 255, 255]) #np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    
    #erosão e dilatação
    erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))

    #erosão de rgb
    red_mask = cv2.erode(red_mask,erodeElement)

    #dilatação de rgb
    red_mask = cv2.erode(red_mask,dilateElement)

    

    # Extraia a região de interesse
    roi = red_mask[cy-1:cy+1, cx-99:cx+100]

    # Calcule o centro dos pixels na ROI
    posicao_centro = np.argwhere(roi == 255)

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in red_contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        area = cv2.contourArea(cnt)
        if area > 400:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv2.putText(belt, str(area), (x, y), 1, 1, (0,255,0))
            moment = cv2.moments(cnt)
            area = moment['m00']
            qcentro_x = int(moment['m10']/area)
            qcentro_y = int(moment['m01']/area)
            cv2.circle(frame, (qcentro_x , qcentro_y), 10, (0,255,0), -1)

            # Coordenadas absolutas dos pontos extremos esquerdo e direito do retângulo em relação ao frame
            ponto_esquerdo_x = x 
            ponto_direito_x = x + w

            # Coordenadas absolutas da lina
            centro_x = 140

            # Calcula a distância entre os pontos extremos e o centro da linha
            distancia_esquerdo = abs(centro_x - ponto_esquerdo_x)
            distancia_direito = abs(centro_x - ponto_direito_x)

            #cv2.putText(frame, f"Distância esquerda: {distancia_esquerdo}, Distância direita: {distancia_direito}", (0, 50), 2, 1, (0, 0, 255), 2)

    # Verifique se há pixels brancos na ROI
    if len(posicao_centro) > 0:
        y,x = roi.shape
        cx_roi = int(x//2)
        cy_roi = int(y//2)
        centro_x = np.mean(posicao_centro[:, 1])
        centro_y = np.mean(posicao_centro[:, 0])

        # Calcule a relação em relação ao centro da imagem
        relacao = (int(centro_x) - cx_roi ) / cx_roi
        cv2.putText(frame, f"Distancia x: {int(centro_x)}, Distancia y: {int(centro_y)}", (0, 50), 2, 1, (0, 0, 255), 2)
        print("Relação em relação ao centro da imagem (-1 a 1):", relacao)
    else:
        print("Nenhum pixel branco na região de interesse.")
    
    cv2.imshow("Red", red_mask)
    cv2.imshow("parte", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break



