import os
import cv2
import numpy as np
import gradio as gr

def clean_speech_bubbles(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. كشف الفقاعات الدائرية/البيضاوية
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0]/16,
                               param1=60, param2=30, minRadius=20, maxRadius=300)
    
    mask = np.zeros_like(gray)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # تبييض داخل الدائرة فقط بدون مسح حدودها
            cv2.circle(mask, center, radius-3, (255), -1)
            
    # 2. كشف الفقاعات المربّعة والشبيهة بالفقاعات
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    binary = cv2.bitwise_not(binary)
    
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 1000 and area < 50000 and w > 30 and h > 30:
            aspect_ratio = float(w)/h
            if 0.5 < aspect_ratio < 2.0:
                roi = gray[y:y+h, x:x+w]
                avg_val = np.mean(roi)
                if avg_val > 220:
                    cv2.rectangle(mask, (x, y), (x+w, y+h), (255), -1)
                    
    # 3. إزالة النص والحفاظ على التفاصيل والحدود
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    return result

# واجهة Gradio
demo = gr.Interface(
    fn=clean_speech_bubbles,
    inputs=gr.Image(type="pil", label="ارفع صفحة المانجا/المانهوا"),
    outputs=gr.Image(type="numpy", label="النتيجة المبيضة"),
    title="أداة تبييض الصفحات التلقائية",
    description="ارفع الصورة هنا وسيم تبييض فقاعات الكلام فوراً"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
