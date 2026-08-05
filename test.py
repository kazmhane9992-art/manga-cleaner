import os
import cv2
import numpy as np
import gradio as gr

def clean_speech_bubbles(image):
    # تحويل الصورة إلى Numpy array
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. كشف التباين العالي (للنصوص والحدود)
    # نستخدم Adaptive Threshold للحصول على دقة عالية جداً لحواف الحروف
    # هذا يساعد على التمييز بين النص على خلفية ملونة والنص على خلفية بيضاء
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # تنظيف الشوائب الصغيرة
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 2. البحث عن الأشكال والمساحات المغلقة
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    
    # 3. تصفية المساحات بناءً على الشكل والمحتوى
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # تشمل الفقاعات فقط (أكبر من حد معين لتجنب مسح الحروف)
        if area > 1000:
            # التحقق من "الاستدارة" أو الشكل البيضاوي
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # إذا كان الشكل قريباً من البيضاوي/الدائري
            if circularity > 0.4:
                # التحقق من متوسط اللون داخل المنطقة (يجب أن يكون فاتحاً)
                x, y, w, h = cv2.boundingRect(cnt)
                roi = gray[y:y+h, x:x+w]
                avg_val = np.mean(roi)
                
                # إذا كانت المنطقة فاتحة جداً (بيضاء)، فهي فقاعة
                if avg_val > 220:
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
            
    # 4. تخمين الخلفية (Inpainting) بشكل ناعم للحفاظ على التدرجات اللونية
    # نستخدم iterations=1 لتنعيم القناع وتجنب الحدود الحادة
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # تطبيق Inpainting لإصلاح النص والحفاظ على الخلفية
    result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result

# واجهة Gradio الاحترافية
demo = gr.Interface(
    fn=clean_speech_bubbles,
    inputs=gr.Image(type="pil", label="ارفع صفحة المانجا/المانهوا"),
    outputs=gr.Image(type="numpy", label="النتيجة المبيضة الذكية"),
    title="أداة تبييض الفقاعات الذكية",
    description="ارفع الصورة هنا وسيتم تبييض فقاعات الكلام فقط والحفاظ على الخلفية الملونة"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
