import os
import cv2
import numpy as np
import gradio as gr

def clean_speech_bubbles(image):
    # تحويل الصورة إلى مصفوفة
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. كشف التباين العالي
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # تنظيف الشوائب الصغيرة
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 2. البحث عن الأشكال والمساحات المغلقة
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. تصفية المساحات وملئها باللون الأبيض
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # تشمل الفقاعات فقط
        if area > 1000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # إذا كان الشكل قريباً من البيضاوي/الدائري
            if circularity > 0.4:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = gray[y:y+h, x:x+w]
                avg_val = np.mean(roi)
                
                # إذا كانت المنطقة فاتحة جداً (فقاعة بيضاء)
                if avg_val > 220:
                    # الحل: التلوين باللون الأبيض الصريح مباشرة بدلاً من inpaint
                    cv2.drawContours(img, [cnt], -1, (255, 255, 255), -1)
            
    return img

# واجهة Gradio
demo = gr.Interface(
    fn=clean_speech_bubbles,
    inputs=gr.Image(type="pil", label="ارفع صفحة المانجا/المانهوا"),
    outputs=gr.Image(type="numpy", label="النتيجة المبيضة الذكية"),
    title="أداة تبييض الفقاعات الذكية",
    description="ارفع الصورة هنا وسيتم تبييض فقاعات الكلام البيضاء بنقاء تام بدون تشوهات."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
