import cv2
import mediapipe as mp
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from datetime import datetime

d = {
    'upward_palm': '\U0001F91A',
    'thumbs_up': '\U0001F44D',
    'victory': '\U0000270C',
    'left_pointing': '\U0001F448',
    'right_pointing': '\U0001F449',
    'upward_pointing': '\U0001F446',
    'downward_pointing': '\U0001F447',
    'left_palm': '\U0001FAF2',
    'right_palm': '\U0001FAF1'
}


def find_coordinates(coordinate_landmark):
    return float(str(coordinate_landmark).split('\n')[0][3:]), float(str(coordinate_landmark).split('\n')[1][3:])

def orientation(coordinate_landmark_0, coordinate_landmark_9):
    x0= float(str(coordinate_landmark_0).split('\n')[0][3:])
    y0= float(str(coordinate_landmark_0).split('\n')[1][3:])
    
    x9= float(str(coordinate_landmark_9).split('\n')[0][3:])
    y9= float(str(coordinate_landmark_9).split('\n')[1][3:])
    
    if abs(x9 - x0) < 0.05:
        m = 1000000000
    else:
        m = abs((y9 - y0)/(x9 - x0))       

    if m>=0 and m<=1:
        if x9 > x0:
            return "Right"
        else:
            return "Left"

    if m>1:
        if y9 < y0:
            return "Up"
        else:
            return "Down"

def check_thumbs_up(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (x3, y3), (x4, y4) = find_coordinates(result.landmark[3]), find_coordinates(result.landmark[4])
    (x5, y5), (x8, y8) = find_coordinates(result.landmark[5]), find_coordinates(result.landmark[8])
    (x9, y9), (x12, y12) = find_coordinates(result.landmark[9]), find_coordinates(result.landmark[12])
    (x13, y13), (x16, y16) = find_coordinates(result.landmark[13]), find_coordinates(result.landmark[16])
    (x17, y17), (x20, y20) = find_coordinates(result.landmark[17]), find_coordinates(result.landmark[20])
    
    if direction == 'Up' or direction == 'Down':
        return False
    
    if y3<y4:
        return False
    
    if direction == 'Left':
        if (x5<x8) and (x9<x12) and (x13<x16) and (x17<x20) and (y4<y5<y9<y13<y17):
            return True
    
    elif direction == 'Right':
        if (x5>x8) and (x9>x12) and (x13>x16) and (x17>x20) and (y4<y5<y9<y13<y17):
            return True
        
    return False

def check_upward_palm(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = find_coordinates(result.landmark[3]), find_coordinates(result.landmark[4])
    (_, y7), (_, y8) = find_coordinates(result.landmark[7]), find_coordinates(result.landmark[8])
    (_, y11), (_, y12) = find_coordinates(result.landmark[11]), find_coordinates(result.landmark[12])
    (_, y15), (_, y16) = find_coordinates(result.landmark[15]), find_coordinates(result.landmark[16])
    (_, y19), (_, y20) = find_coordinates(result.landmark[19]), find_coordinates(result.landmark[20])
    
    if check_thumbs_up(result):
        return False
    
    if direction == 'Down' or direction == 'Left' or direction == 'Right':
        return False
    
    if (y4<y3) and (y8<y7) and (y12<y11) and (y16<y15) and (y20<y19) and (y4>y8) and (y4>y12) and (y4>y16) and (y4>y20):
        return True
        
    return False

def check_victory(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = find_coordinates(result.landmark[3]), find_coordinates(result.landmark[4])
    (_, y7), (_, y8) = find_coordinates(result.landmark[7]), find_coordinates(result.landmark[8])
    (_, y11), (_, y12) = find_coordinates(result.landmark[11]), find_coordinates(result.landmark[12])
    (_, y15), (_, y16) = find_coordinates(result.landmark[15]), find_coordinates(result.landmark[16])
    (_, y19), (_, y20) = find_coordinates(result.landmark[19]), find_coordinates(result.landmark[20])
    (_, y13), (_, y17) = find_coordinates(result.landmark[13]), find_coordinates(result.landmark[17])
    (_, y14), (_, y18) = find_coordinates(result.landmark[14]), find_coordinates(result.landmark[18])
    
    if direction == 'Down' or direction == 'Right' or direction == 'Left':
        return False
    
    if (y7>y8) and (y11>y12) and (y16>y15) and (y20>y19) and (y3>y4) and (y4>y14) and (y4>y18):
        return True
    
    return False

def check_left_pointing(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = find_coordinates(result.landmark[3]), find_coordinates(result.landmark[4])
    (x8, y8), (x7, _) = find_coordinates(result.landmark[8]), find_coordinates(result.landmark[7])
    (x12, y12) = find_coordinates(result.landmark[12])
    (x16, y16) = find_coordinates(result.landmark[16])
    (x20, y20) = find_coordinates(result.landmark[20])
    (x6, _), (x10, _) = find_coordinates(result.landmark[6]), find_coordinates(result.landmark[10])
    (x14, _), (x18, _) = find_coordinates(result.landmark[14]), find_coordinates(result.landmark[18])
    
    if direction == 'Down' or direction == 'Right' or direction == 'Up':
        return False
    
    if (y3>y4) and (y4<y8<y12<y16<y20) and (x6>x7>x8) and (x12>x10) and (x16>x14) and (x20>x18):
        return True
    
    return False

def check_right_pointing(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = find_coordinates(result.landmark[3]), find_coordinates(result.landmark[4])
    (x8, y8), (x7, _) = find_coordinates(result.landmark[8]), find_coordinates(result.landmark[7])
    (x12, y12) = find_coordinates(result.landmark[12])
    (x16, y16) = find_coordinates(result.landmark[16])
    (x20, y20) = find_coordinates(result.landmark[20])
    (x6, _), (x10, _) = find_coordinates(result.landmark[6]), find_coordinates(result.landmark[10])
    (x14, _), (x18, _) = find_coordinates(result.landmark[14]), find_coordinates(result.landmark[18])
    
    if direction == 'Down' or direction == 'Left' or direction == 'Up':
        return False
    
    if (y3>y4) and (y4<y8<y12<y16<y20) and (x6<x7<x8) and (x12<x10) and (x16<x14) and (x20<x18):
        return True
    
    return False

def check_upward_pointing(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = find_coordinates(result.landmark[3]), find_coordinates(result.landmark[4])
    (x7, y7), (x8, y8) = find_coordinates(result.landmark[7]), find_coordinates(result.landmark[8])
    (x9, y9), (x12, y12) = find_coordinates(result.landmark[9]), find_coordinates(result.landmark[12])
    (x13, y13), (x16, y16) = find_coordinates(result.landmark[13]), find_coordinates(result.landmark[16])
    (x17, y17), (x20, y20) = find_coordinates(result.landmark[17]), find_coordinates(result.landmark[20])
    
    if direction == 'Down' or direction == 'Left' or direction == 'Right':
        return False
    
    if (y3>y4) and (y7>y8) and (y12>y9) and (y16>y13) and (y20>y17) and ((x7>x9>x13>x17) or (x7<x9<x13<x17)):
        return True
    
    return False

def check_downward_pointing(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    (_, y3), (_, y4) = find_coordinates(result.landmark[3]), find_coordinates(result.landmark[4])
    (x7, y7), (_, y8) = find_coordinates(result.landmark[7]), find_coordinates(result.landmark[8])
    (x9, _), (_, y12) = find_coordinates(result.landmark[9]), find_coordinates(result.landmark[12])
    (x13, _), (_, y16) = find_coordinates(result.landmark[13]), find_coordinates(result.landmark[16])
    (x17, _), (_, y20) = find_coordinates(result.landmark[17]), find_coordinates(result.landmark[20])
    (_, y14), (_, y10) = find_coordinates(result.landmark[14]), find_coordinates(result.landmark[10])
    (_, y18) = find_coordinates(result.landmark[18])
    
    if direction == 'Up' or direction == 'Left' or direction == 'Right':
        return False
    
    if (y3<y4) and (y7<y8) and (y12<y10) and (y16<y14) and (y20<y18) and ((x7>x9>x13>x17) or (x7<x9<x13<x17)):
        return True
    
    return False

def check_left_palm(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    #return (direction == 'Left')
    (_, y3), (_, y4) = find_coordinates(result.landmark[3]), find_coordinates(result.landmark[4])
    (x8, y8), (x7, _) = find_coordinates(result.landmark[8]), find_coordinates(result.landmark[7])
    (x12, y12), (x11, y11) = find_coordinates(result.landmark[12]), find_coordinates(result.landmark[11])
    (x16, y16), (x15, y15) = find_coordinates(result.landmark[16]), find_coordinates(result.landmark[15])
    (x20, y20), (x19, y19) = find_coordinates(result.landmark[20]), find_coordinates(result.landmark[19])
    (x6, _), (x10, _) = find_coordinates(result.landmark[6]), find_coordinates(result.landmark[10])
    (x14, _), (x18, _) = find_coordinates(result.landmark[14]), find_coordinates(result.landmark[18])
    
    if direction == 'Down' or direction == 'Right' or direction == 'Up':
        return False
    
    if (y3>y4) and (y4<y8<y12<y16<y20) and (x7>x8) and (x11>x12) and (x15>x16) and (x19>x20):
        return True
    
    return False

def check_right_palm(result):
    direction = orientation(result.landmark[0], result.landmark[9])
    #return (direction == 'Left')
    (_, y3), (_, y4) = find_coordinates(result.landmark[3]), find_coordinates(result.landmark[4])
    (x8, y8), (x7, _) = find_coordinates(result.landmark[8]), find_coordinates(result.landmark[7])
    (x12, y12), (x11, y11) = find_coordinates(result.landmark[12]), find_coordinates(result.landmark[11])
    (x16, y16), (x15, y15) = find_coordinates(result.landmark[16]), find_coordinates(result.landmark[15])
    (x20, y20), (x19, y19) = find_coordinates(result.landmark[20]), find_coordinates(result.landmark[19])
    (x6, _), (x10, _) = find_coordinates(result.landmark[6]), find_coordinates(result.landmark[10])
    (x14, _), (x18, _) = find_coordinates(result.landmark[14]), find_coordinates(result.landmark[18])
    
    if direction == 'Down' or direction == 'Left' or direction == 'Up':
        return False
    
    if (y3>y4) and (y4<y8<y12<y16<y20) and (x7<x8) and (x11<x12) and (x15<x16) and (x19<x20):
        return True
    
    return False

def get_shuffled_dictionary():
    items = list(d.items())
    random.shuffle(items)
    shuffled_dict = dict(items)
    return shuffled_dict

def check_actions(string, direction):
    if string == 'upward_palm':
        return check_upward_palm(direction)
    elif string == 'thumbs_up':
        return check_thumbs_up(direction)
    elif string == 'victory':
        return check_victory(direction)
    elif string == 'left_pointing':
        return check_left_pointing(direction)
    elif string == 'right_pointing':
        return check_right_pointing(direction)
    elif string == 'upward_pointing':
        return check_upward_pointing(direction)
    elif string == 'downward_pointing':
        return check_downward_pointing(direction)
    elif string == 'left_palm':
        return check_left_palm(direction)
    elif string == 'right_palm':
        return check_right_palm(direction)
    else:
        return False

frameWidth = 720
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

seq_dict = get_shuffled_dictionary()
start_time = 0

while True:
    seq_dict = get_shuffled_dictionary()
    success, img = cap.read()
    img= cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img.shape
    fontpath = r"./Font.ttf"
    fontpath_bold = r"./Font_Bold.ttf"
    game_display = np.zeros((h, w, 3),np.uint8)*255
    font = ImageFont.truetype(fontpath, 20)
    font_bold = ImageFont.truetype(fontpath_bold, 30)
    img_pil = Image.fromarray(game_display)
    draw = ImageDraw.Draw(img_pil)
    draw.text((250, 130), 'Start Game', embedded_color=True, font = font_bold, fill=(255, 255, 255))
    draw.text((210, 180), 'Press Enter to Start Game', embedded_color=True, font = font, fill=(102, 0, 204))
    draw.text((250, 230), 'End Game', embedded_color=True, font = font_bold, fill=(255, 255, 255))
    draw.text((210, 280), 'Press Esc to End the Game', embedded_color=True, font = font, fill=(102, 0, 204))
    seq = np.array(img_pil)
    img[:, :] = seq
    total_score = 0

    if cv2.waitKey(1) == 13:
        start_time = time.time()
        while True:
            success, img = cap.read()
            img= cv2.flip(img,1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            sequence = ''.join(seq_dict.values())
            string_sequence = list(seq_dict.keys())
            
            fontpath = r"./NotoEmoji-VariableFont_wght.ttf"
            emoji_sequence = np.zeros((50, 460, 3),np.uint8)
            font = ImageFont.truetype(fontpath, 40)
            img_pil = Image.fromarray(emoji_sequence)
            draw = ImageDraw.Draw(img_pil)
            draw.text((0, 0), sequence, embedded_color=True, font = font, fill=(0, 255, 255))
            seq = np.array(img_pil)
            
            img[:50, 80:540] = seq
            
            elapsed_time = time.time() - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            cv2.putText(img, 'Time: '+formatted_time, (380, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            cv2.putText(img, 'Signs Completed: ' + str(total_score), (380, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
            
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                    direction = results.multi_hand_landmarks[-1]
                    if check_actions(string_sequence[total_score], direction):
                        total_score += 1
                        cv2.imshow('image', img)
                        if total_score == 9:
                            break
            
            cv2.imshow('image', img)
            if cv2.waitKey(1)==27:
                break
            
            if total_score == 9:
                break
            
    time_taken = round(time.time() - start_time, 0)
    while total_score == 9:
        success, img = cap.read()
        img= cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        score_display = np.zeros((h, w, 3),np.uint8)*255
        img_pil = Image.fromarray(score_display)
        draw = ImageDraw.Draw(img_pil)
        draw.text((230, 130), 'Game Over', embedded_color=True, font = font_bold, fill=(255, 255, 255))
        draw.text((200, 190), 'Time taken: '+str(time_taken), embedded_color=True, font = font_bold, fill=(102, 255, 204))
        draw.text((170, 250), 'Press Esc for Main Menu', embedded_color=True, font = font_bold, fill=(255, 255, 255))
        seq = np.array(img_pil)
        img[:, :] = seq
        
        cv2.imshow('image', img)
        if cv2.waitKey(1)==27:
                break
            
    cv2.imshow('image', img)
    if cv2.waitKey(1)==27:
            break