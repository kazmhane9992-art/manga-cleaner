import cv2
import numpy as np
from PIL import Image
import gradio as gr

def clean_speech_bubbles(input_img):
    if input_img is None:
        return None
    
    # تحويل الصورة إلى صيغة OpenCV
    image = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

    # معالجة الصورة وتحديد الفقاعات
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            cv2.drawContours(image, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)

    # إرجاع الصورة المبيضة بصيغة RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# بناء واجهة الموقع
demo = gr.Interface(
    fn=clean_speech_bubbles,
    inputs=gr.Image(type="pil", label="ارفع صفحة المانجا/المانهوا"),
    outputs=gr.Image(type="numpy", label="النتيجة المبيضة"),
    title="أداة تبييض الصفحات التلقائية",
    description="ارفع الصورة هنا وسيتم تبييض فقاعات الكلام فوراً."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
