import os
import cv2
import numpy as np
import gradio as gr

def clean_speech_bubbles(image):
    # تحويل الصورة إلى مصفوفة
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. كشف التباين العالي (النصوص داخل الفقاعات أو فوق المؤثرات)
    # استخدام Adaptive Threshold للحصول على دقة عالية جداً لحواف الحروف
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY_INV, 15, 10)
    
    # 2. تنظيف الشوائب وتركيز القناع على النصوص فقط دون مسح المؤثرات الكبيرة
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 3. تمديد طفيف جداً للقناع لضمان تغطية حدود الحروف بالكامل
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 4. تخمين الخلفية وإعادة رسمها (Inpainting)
    # هذه التقنية تأخذ الألوان المحيطة بالنص وتدمجها وتخمن ما وراءه لتعبئته بنفس الخلفية (سواء كانت زرقاء، شفافة، أو متدرجة) بدلاً من اللون الأبيض الصريح
    result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result

# واجهة Gradio الاحترافية
demo = gr.Interface(
    fn=clean_speech_bubbles,
    inputs=gr.Image(type="pil", label="ارفع صفحة المانجا/المانهوا"),
    outputs=gr.Image(type="numpy", label="النتيجة الذكية المخمنة للخلفية"),
    title="أداة تبييض وتخمين خلفيات المانجا الذكية",
    description="ارفع الصورة هنا وسيتم إزالة النص وتخمين الخلفية والمؤثرات بدقة عالية"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
