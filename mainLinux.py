import pyautogui as pag
import pyscreeze
import pytesseract
import numpy as np
import cv2
import keyboard
import imutils
import argparse
import easyocr
from multiprocessing import pool
from time import sleep, perf_counter

"""
    This program plays the 1-50 game at https://www.arealme.com/1-to-50/en/
    It scans the playing area and finds all numbers, then it clicks all the number and scans again for the other set of 25 numbers.
    Only tested with a full screen chrome (F11) and the game zoomed to fill the screen.
    Current record is 2.306s
    Brute force record is 0.601s
"""

# Switch to chrome window, then take a screenshot and find the position of each number.
def scanNumbers(debug=False, blackBackground=False):
    #window = getWindow()
    # Give some time for the OS to switch window
    sleep(0.1)
    #imageRGB = pag.screenshot()
    imageRGB = pyscreeze.screenshot()
    imageArray = np.array(imageRGB)
    #print(imageArray)
    if blackBackground:
        imageArray[imageArray<=100] = 255
    #print(imageArray)
    image = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((9,9)))
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buttons = {}
    imageOCR = cv2.bitwise_not(image)
    _, imageOCR = cv2.threshold(imageOCR, 100, 255, cv2.THRESH_BINARY)
    if debug:
        # Show images of what is happening.
        imageBGR = cv2.cvtColor(imageArray, cv2.COLOR_RGB2BGR)
        cv2.imshow("image",imutils.resize(imageBGR, height=np.shape(imageBGR)[0]//2))
        cv2.imshow("thresh", imutils.resize(thresh, height=np.shape(thresh)[0]//2))
        cv2.imshow("opening", imutils.resize(opening, height=np.shape(opening)[0]//2))
        cv2.drawContours(imageBGR, contours, -1, (0,255,0))
        cv2.imshow("contours",imutils.resize(imageBGR, height=np.shape(imageBGR)[0]//2))
    for contour in contours:
        # Use tesseract to scan each square and find what number is in it
        startTime = perf_counter()
        x, y, w, h = cv2.boundingRect(contour)
        roi = imageOCR[y:y+h, x:x+w]
        result = pytesseract.image_to_string(roi, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        if len(result)==0:
            result = '9'
        else:
            result = result[0:-1]
        #if int(result) == 4 and not np.sum(roi) < 14300630:
        #    result = "7"
        buttons[int(result)] = (x,y)
        endTime = perf_counter()
        if debug:
            # Show the image being processed and the resut it gives.
            print(result)
            cv2.imshow('roi', roi)
            cv2.waitKey()
            print(f"Time to detect {result} was {endTime-startTime}s")
    return buttons

# Use tesseract to scan each square and find what number is in it
def numberDetect(img):
    result = pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    # Hard coding
    if len(result)==0:
        result = '9'
    else:
        result = result[0:-1]
    return int(result)

def parallelScanNumbers(debug=False, blackBackground=False):
    #window = getWindow()
    # Give some time for the OS to switch window
    sleep(0.1)
    #imageRGB = pag.screenshot()
    imageRGB = pyscreeze.screenshot()
    imageArray = np.array(imageRGB)
    if blackBackground:
        imageArray[imageArray<=100] = 255
    image = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((9,9)))
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imageOCR = cv2.bitwise_not(image)
    _, imageOCR = cv2.threshold(imageOCR, 100, 255, cv2.THRESH_BINARY)
    if debug:
        # Show images of what is happening.
        imageBGR = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("image",imutils.resize(imageBGR, height=np.shape(imageBGR)[0]//2))
        cv2.imshow("thresh", imutils.resize(thresh, height=np.shape(thresh)[0]//2))
        cv2.imshow("opening", imutils.resize(opening, height=np.shape(opening)[0]//2))
        cv2.drawContours(imageBGR, contours, -1, (0,255,0))
        cv2.imshow("contours",imutils.resize(imageBGR, height=np.shape(imageBGR)[0]//2))
    boundingRects = [cv2.boundingRect(i) for i in contours]
    croppedImages = [imageOCR[i[1]:i[1]+i[3], i[0]:i[0]+i[2]] for i in boundingRects]
    with pool.Pool(numProcessors) as processor:
        # Run the digit detection in parallel
        numbers = list(processor.map(numberDetect, croppedImages))
    return {numbers[i]:boundingRects[i] for i in range(len(numbers))}

def main(debug=False, blackBackground = False):
    startTime = perf_counter()
    buttons = scanNumbers(debug, blackBackground)
    endTime = perf_counter()
    print(buttons)
    print(f'Time to find buttons: {endTime-startTime}s')
    for i in range(1,26):
        try:
            #pag.click(buttons[i][0]+25,buttons[i][1]+25, clicks=10)
            pag.leftClick(buttons[i][0]+25,buttons[i][1]+25)
        except KeyError:
            # Hard coding
            print(f'Exception at index {i}')
            if i==9: i=0
            pag.leftClick(buttons[i][0]+25,buttons[i][1]+25)
    sleep(0.3)
    startTime = perf_counter()
    buttons2 = scanNumbers(blackBackground=blackBackground)
    endTime = perf_counter()
    print(buttons2)
    print(f'Time to find buttons2: {endTime-startTime}s')
    for i in range(26,51):
        try:
            pag.leftClick(buttons2[i][0]+25,buttons2[i][1]+25)
        except KeyError:
            # Hard coding
            print(f'Exception at index {i}')
            if i==37: i=3
            elif i==50: i=90
            elif i==41: i=4
            pag.leftClick(buttons2[i][0]+25,buttons2[i][1]+25)

def parallelMain(debug=False, blackBackground = False):
    startTime = perf_counter()
    buttons = parallelScanNumbers(debug, blackBackground)
    endTime = perf_counter()
    #print(buttons)
    print(f'Time to find buttons: {endTime-startTime}s')
    for i in range(1,26):
        pag.leftClick(buttons[i][0]+50,buttons[i][1]+50)
    sleep(0.12)
    startTime = perf_counter()
    buttons2 = parallelScanNumbers(blackBackground=blackBackground)
    endTime = perf_counter()
    #print(buttons2)
    print(f'Time to find buttons2: {endTime-startTime}s')
    for i in range(26,51):
        try:
            pag.leftClick(buttons2[i][0]+25,buttons2[i][1]+25)
        except KeyError:
            # Hard coding
            print(f'Exception at index {i}')
            if i==37: i=3
            elif i==50: i=90
            elif i==41: i=4

def bruteforce(debug=False, blackBackground = False):
    buttons = parallelScanNumbers(debug, blackBackground)
    for _ in range(26):
        for i in range(1,26):
            pag.leftClick(buttons[i][0]+50,buttons[i][1]+50)
            print(f'Exception at index {i}')

def runFunctions(args):
    if args.brute_force:
        bruteforce(args.debug)
    elif args.parallel:
        parallelMain(args.debug)
    else:
        main(args.debug)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--speed", type=float,
                        help="Adjusting clicking speed of pyautogui. Default is 0.0001s")
    parser.add_argument("-c", "--cpu", type=int,
                        help="Number of CPUs to use when running in parallel. Default is 5")
    parser.add_argument("-w", "--window", type=str,
                        help="Name of window where the game is. Default is '1 to 50 - How Fast Can You Click From ONE to FIFTY? - Google Chrome'")
    # This ended up not being what I wanted
    #parser.add_argument("-p", "--parallel", type=argparse.BooleanOptionalAction,
    #                    help="Runs digit detection in parallel")
    parser.add_argument("-d", "--debug", nargs='?', default=False, const=True,
                        help="Enter debug mode. Shows pre-processing images. Press any key to continue")
    parser.add_argument("-p", "--parallel", nargs='?', default=False, const=True,
                        help="Runs digit detection in parallel")
    parser.add_argument("-b", "--brute-force", nargs='?', default=False, const=True,
                        help="Runs second half of the game by clicking until it works. Overrides parallel argument")
    parser.add_argument("-bb", "--black-background", nargs='?', default=False, const=True,
                        help="Set this if your browser is in dark mode and your background is black")
    args = parser.parse_args()
    pag.PAUSE = args.speed or 0.0001
    numProcessors = args.cpu or 5
    windowName = args.window or "1 to 50 - How Fast Can You Click From ONE to FIFTY? - Google Chrome"
    #keyboard.add_hotkey("ctrl+alt+shift+s", runFunctions, args=(args,))
    keyboard.wait("ctrl+alt+s")
    if args.brute_force:
        bruteforce(args.debug, args.black_background)
    elif args.parallel:
        parallelMain(args.debug, args.black_background)
    else:
        main(args.debug, args.black_background)
    #getWindow()
    cv2.waitKey()
