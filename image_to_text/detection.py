import cv2
import pytesseract
import numpy as np

def capture(sample=None):

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("take picture")

    img_counter = 0
    if sample is None:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.jpg".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img = cv2.imread(img_name)
                img = cv2.resize(img, (400, 400))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                config = "--psm 3"
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                text = pytesseract.image_to_string(gray, config=config)
                print(text)
                img_counter += 1
                
    elif sample is not None:
        img = cv2.imread(sample)
        img = cv2.resize(img, (400, 400))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        config = "--psm 3"

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text = pytesseract.image_to_string(gray, config=config)
        print(text)


    cam.release()
    cv2.destroyAllWindows()
    return text


if __name__=='__main__':
    capture("sample.jpg")
