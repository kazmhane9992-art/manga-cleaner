import os
import cv2
import numpy as np
import gradio as gr

def clean_speech_bubbles(image):
    # تحويل الصورة إلى مصفوفة
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. كشف دقيق جداً للنصوص السوداء داخل الفقاعات الفاتحة
    # نستخدم thresholding مضبوط خصيصاً لالتقاط الحروف الغامقة مهما كانت دقيقة
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 2. توسيع حواف النص (Dilation) لضمان مسح كل آثار الحروف القديمة تماماً وجذرها
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(thresh, kernel, iterations=2)
    
    # 3. إزالة أي شوائب صغيرة لا تمثل حروفاً
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 4. إعادة بناء الخلفية (Inpainting) بذكاء لدمج النص المسحوح مع لون الفقاعة الأساسي
    result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result

# واجهة Gradio
demo = gr.Interface(
    fn=clean_speech_bubbles,
    inputs=gr.Image(type="pil", label="ارفع صفحة المانجا/المانهوا"),
    outputs=gr.Image(type="numpy", label="النتيجة المبيضة"),
    title="أداة تبييض فقاعات المانجا",
    description="ارفع الصورة هنا لتتم إزالة الكلام تماماً وبشكل ناعم"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
