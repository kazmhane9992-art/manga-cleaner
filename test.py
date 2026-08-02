import cv2
import numpy as np

def clean_speech_bubbles(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. عزل المساحات البيضاء الساطعة جداً (الفقاعات)
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # 2. إيجاد حدود الأشكال المغلقة في الصورة
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # فلترة المساحات: استهداف الفقاعات الكبيرة وتجاهل المؤثرات الصغيرة
        if area > 800:  
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
    # 3. تبييض داخل الفقاعات المحددة فقط وتفريغ النص
    result = img.copy()
    result[mask == 255] = [255, 255, 255]
    
    return result    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
