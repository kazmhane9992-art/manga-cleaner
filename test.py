import os
import cv2
import numpy as np
import gradio as gr

def clean_speech_bubbles(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. تحديد المناطق البيضاء الساطعة جداً (خلفية الفقاعات)
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # 2. البحث عن الأشكال والمساحات المغلقة
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    
    # 3. تصفية المساحات: استهداف فقاعات الكلام (حسب المساحة) وتجاهل ملامح الوجه والمؤثرات
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # تشمل الفقاعات فقط (أكبر من 2000 بكسل) لتجنب مسح الفم أو العينين
        if area > 2000:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
    # 4. تقليص قناع التبييض بمقدار 2 بكسل للحفاظ على حدود الفقاعة السوداء
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # 5. تطبيق التبييض السريع
    result = img.copy()
    result[mask == 255] = [255, 255, 255]
    
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
