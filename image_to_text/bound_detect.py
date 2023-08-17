import cv2
import pytesseract
import streamlit as st

pytesseract.pytesseract.tesseract_cmd = r'assets\Tesseract-OCR\tesseract.exe'

def capture(sample=None):
    
    if sample is None:

        video = cv2.VideoCapture(0)
        #video.set(cv2.CAP_PROP_FPS, 25)
        #video.set(3, 640)
        #video.set(1920, 1080)
        ocr_result = []
        while True:
            k = cv2.waitKey(1)
            success, image = video.read()
            if not success:
                print("failed to grab frame")
                break

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7,7), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
            dilate = cv2.dilate(thresh, kernel, iterations=1)

            # find the countours
            cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
            
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                img = cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 2)
                if k%256 == 32:
                    ocr_result2 = pytesseract.image_to_string(img)
                    ocr_result.append(ocr_result2)
                    break

            cv2.imshow("Optical Character Recogntion", img)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
        ocr_result = [x.replace("\n", " ") for x in ocr_result]
        video.release()
        cv2.destroyAllWindows()
            
    elif sample is not None:
        image = cv2.imdecode(sample, 1)
        #image = cv2.imread(sample)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
        dilate = cv2.dilate(thresh, kernel, iterations=1)

        # find the countours
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            roi = cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 2)
            ocr_result = pytesseract.image_to_string(roi)
            break
        
    return ocr_result
        

if __name__=="__main__":
    capture()
