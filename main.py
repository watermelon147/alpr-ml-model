import cv2
from ultralytics import YOLO
import easyocr

if __name__ == '__main__':
    
    model_path = r'C:\Users\samar\Documents\num-plt-detection-ml-model\runs\detect\ALPR_Project\plate_detector\weights\best.pt'
    model = YOLO(model_path)

  
    reader = easyocr.Reader(['en'], gpu=True)

  
    image_path = r'C:\Users\samar\Documents\num-plt-detection-ml-model\test-img\test_car3.jpg' 
    img = cv2.imread(image_path)

   
    results = model.predict(source=img, conf=0.5)

 
    for result in results:
        boxes = result.boxes
        for box in boxes:
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
           
            plate_crop = img[y1:y2, x1:x2]
            
           
            ocr_result = reader.readtext(plate_crop)
            
           
            if ocr_result:
               
                text = ocr_result[0][1]
                print(f"\n--- SUCCESS! Detected Plate Text: {text} ---\n")
                
               
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    
    cv2.imshow("ALPR System", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
