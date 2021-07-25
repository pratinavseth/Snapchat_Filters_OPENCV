import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os


def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

specs_ori = cv2.imread('Daco_5147210.png', -1)
cigar_ori = cv2.imread('cigar.png', -1)
mus_ori = cv2.imread('mustache.png', -1)
SCIENCE = cv2.imread('Daco_4047678.png',-1)
specs_prof = cv2.imread('Daco_13494.png',-1)
apron_img = cv2.imread('Daco_4797160.png',-1)
ieee_img=cv2.imread('ieeewhiteblack.png',-1)
tc_img = cv2.imread('logo.png',-1)

cap = cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,480)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
lsImg = os.listdir("/home/ptnv-s/PycharmProjects/IEEE/img")
print(lsImg)
imgls = []

for imgPath in lsImg:
    img = cv2.imread(f'/home/ptnv-s/PycharmProjects/IEEE/img/{imgPath}')
    imgls.append(img)
print(imgls)

imgIndex = 0

while True:
    cap.set(cv2.CAP_PROP_FPS, 10)
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.2, 5, 0, (120, 120), (350, 350))
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            m_symin = int(y + 1.4 * h / 5)
            m_symax = int(y + 3.5 * h / 5)
            mm_glass = m_symax - m_symin

            m_glass_roi_color = img[m_symin:m_symax, round(x + w / 10):round(x + w - w / 10)]
            specsm = cv2.resize(specs_prof, (round(8 * w / 10), mm_glass), interpolation=cv2.INTER_CUBIC)
            transparentOverlay(m_glass_roi_color, specsm)

            glass_symin = int(y + 1.2 * h / 5)
            glass_symax = int(y + 2.8 * h / 5)
            sh_glass = glass_symax - glass_symin

            face_glass_roi_color = img[glass_symin:glass_symax, x:x + w]
            specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
            transparentOverlay(face_glass_roi_color, specs)

            ap_symin = int(y+0.999*h)
            ap_symax = int(480.0)
            ap_glass = ap_symax - ap_symin

            ap_glass_roi_color = img[ap_symin:ap_symax, int(x-0.4*w):int(x + 1.6*w)]
            apron = cv2.resize(apron_img, (int(2.0*w), ap_glass), interpolation=cv2.INTER_CUBIC)
            transparentOverlay(ap_glass_roi_color, apron)


    sci_img = cv2.resize(ieee_img, (150, 150), interpolation=cv2.INTER_CUBIC)
    science_img = cv2.resize(SCIENCE, (100, 100), interpolation=cv2.INTER_CUBIC)
    tc1_img = cv2.resize(tc_img, (100, 100), interpolation=cv2.INTER_CUBIC)
    imgOut = segmentor.removeBG(img, imgls[imgIndex], threshold=0.3)
    imgOut[0:150, 0:150, :] = sci_img[0:150, 0:150, 0:3]
    imgOut[150:250, 0:100, :] = tc1_img[0:100, 0:100, 0:3]
    imgOut[0:100, 150:250, :] = science_img[0:100, 0:100, 0:3]
    imgStacked = cvzone.stackImages([img, imgOut], 2,1)
    _, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))
    print(imgIndex)
    cv2.imshow("Image",imgStacked)

    key = cv2.waitKey(1)

    if(key==ord('a')):
        if(imgIndex>0):
            imgIndex-=1
    elif (key == ord('d')):
        if(imgIndex<len(lsImg)-1):
            imgIndex += 1
    elif (key == ord('q')):
        break