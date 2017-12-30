import cv2
import numpy as np
# import matplotlib.pyplot as plt
import glob
from urllib.request import urlopen


def pers_transform(w, h, approx, img):
    # print approx
    cood = np.zeros((4, 2), dtype="float32")
    s = approx.sum(axis=1)
    tl = approx[np.argmin(s)]  # top left pt will have the smallest sum
    br = approx[np.argmax(s)]  # bottom right will have the largest sum

    d = np.diff(approx, axis=1)
    tr = approx[np.argmin(d)]  # top right has the smallest diff
    bl = approx[np.argmax(d)]  # bottom left has the largest diff

    if w <= 0.8 * h:  # vertically
        cood[0] = tl
        cood[1] = tr
        cood[2] = br
        cood[3] = bl
    if w >= 1.2 * h:  # horizontally
        cood[0] = bl
        cood[1] = tl
        cood[2] = tr
        cood[3] = br
    if w > 0.8 * h and w < 1.2 * h:
        if approx[1][1] <= approx[3][1]:
            approx = sorted(approx, key=lambda x: x[1])
            cood[0] = approx[2]
            cood[1] = approx[3]
            cood[2] = approx[1]
            cood[3] = approx[0]
            # print "HELLO1"
        if approx[1][1] > approx[3][1]:
            approx = sorted(approx, key=lambda x: x[1])
            # print approx
            cood[0] = approx[3]
            cood[1] = approx[2]
            cood[2] = approx[0]
            cood[3] = approx[1]
            # print "HI"

    maxWidth = 200
    maxHeight = 300
    # print cood
    # r=np.int0(r)
    # cv2.drawContours(img,contours,0,(0,255,0),2)
    # cv2.imshow("jbj",img)
    h = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    ##print h

    transform = cv2.getPerspectiveTransform(cood, h)
    # print transform
    warp = cv2.warpPerspective(img, transform, (maxWidth, maxHeight))
    return warp


def suit_detection(thresh, contours):
    m = 0;
    con = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > m:
            con = i;
            m = cv2.contourArea(contours[i])

    x1, y1, w1, h1 = cv2.boundingRect(contours[con])

    ##    cv2.imshow('nn',thresh)
    th_suit = thresh[y1:y1 + h1, x1:x1 + w1]
    th_suit_zoom = cv2.resize(th_suit, (70, 100), 0, 0)
    # cv2.imshow('nvn',th_suit_zoom)
    diff = 10000
    suit_names = ['clubs', 'Diamonds', 'Hearts', 'Spades']
    images = glob.glob("test/*.png")

    suit_img = []
    for i in images:
        test_image = cv2.imread(i)
        suit_img.append(test_image)
    for i in range(len(suit_img)):
        suit_img_gray = cv2.cvtColor(suit_img[i], cv2.COLOR_BGR2GRAY)
        _, thresh5 = cv2.threshold(suit_img_gray, 150, 255, cv2.THRESH_BINARY)
        diff_img = cv2.absdiff(th_suit_zoom, thresh5)
        suit_diff = int(np.sum(diff_img) / 255)
        # print suit_diff
        if suit_diff < diff:
            diff = suit_diff
            name = suit_names[i]
            # print 'HI'
    return name


def rank_detection(thresh, contours):
    m1 = 0;
    con1 = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > m1:
            con1 = i;
            m1 = cv2.contourArea(contours[i])

    x2, y2, w2, h2 = cv2.boundingRect(contours[con1])
    # cv2.rectangle(rank_zoom,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)

    th_rank = thresh[y2:y2 + h2, x2:x2 + w2]
    th_rank_zoom = cv2.resize(th_rank, (70, 125), 0, 0)
    # cv2.imshow('jj',th_rank_zoom)
    # cv2.imwrite('Six.jpg',th_rank_zoom)
    diff = 10000
    rank_names = ['ACE', 'Eight', 'Five', 'Four', 'Jack', 'King', 'Nine', 'Queen', 'Seven', 'Six', 'Ten', 'Three',
                  'Two']
    rank_images = glob.glob("test2/*.png")

    rank_img = []
    for i in rank_images:
        test_image = cv2.imread(i)
        rank_img.append(test_image)
    for i in range(len(rank_img)):
        rank_img_gray = cv2.cvtColor(rank_img[i], cv2.COLOR_BGR2GRAY)
        _, thresh6 = cv2.threshold(rank_img_gray, 150, 255, cv2.THRESH_BINARY)
        diff_img = cv2.absdiff(th_rank_zoom, thresh6)
        rank_diff = int(np.sum(diff_img) / 255)
        # print rank_diff
        if rank_diff < diff:
            diff = rank_diff
            name1 = rank_names[i]
            # print 'HI'
    return name1


def zoom(image):
    image_zoom = cv2.resize(image, (0, 0), fx=4, fy=3)
    image_zoom_gray = cv2.cvtColor(image_zoom, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(image_zoom_gray, 150, 255, cv2.THRESH_BINARY_INV)  # suit thresholded
    # img_thresh=cv2.adaptiveThreshold(image_zoom_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,2)
    _, img_contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print len(img_contours)
    return img_thresh, img_contours


def detect(img):
    # img=cv2.imread('card29.jpg')

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img1, (13, 11), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = []
    mar = 120000;
    pos = -1
    for i in range(len(contours)):
        peri = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.01 * peri, True)
        ar = cv2.contourArea(contours[i])
        if ar >= 15000 and ar <= 120000 and len(approx) == 4:
            if ar < mar:
                mar = ar
                pos = i
                app = approx

    x, y, w, h = cv2.boundingRect(contours[pos])

    app.resize((4, 2))

    app = np.float32(app)

    warp = pers_transform(w, h, app, img)  # transfom the card
    # cv2.imshow('kk',warp)
    maxWidth = 200
    maxHeight = 300
    rs = warp[0:int(maxHeight / 3), 0:int(maxWidth / 6)]

    rank = rs[0:int(maxHeight / 6.3), 0:int(maxWidth / 5)]  # rank original
    suit = rs[int(maxHeight / 6.3 - 1.5):, 0:int(maxWidth / 5)]  # suit original

    suit_thresh, suit_contour = zoom(suit)

    rank_thresh, rank_contour = zoom(rank)

    suit_name_detected = suit_detection(suit_thresh, suit_contour)

    rank_name_detected = rank_detection(rank_thresh, rank_contour)

    return rank_name_detected + " of " + suit_name_detected


##    cv2.imshow('hbh',img)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
def get_image(url):
    resp = urlopen('https://firebasestorage.googleapis.com/v0/b/fir-test-1138e.appspot.com/o/chat_photos%2Fimage%3A292882?alt=media&token=6aa6f61d-4859-4a1b-987f-36b392ca6d03')
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, (270, 480), interpolation=cv2.INTER_AREA)
    return detect(resized)
