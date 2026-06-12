import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from dotenv import load_dotenv
from google import genai              # Dùng thư viện mới giống PhoBot
from google.genai import types        # Dùng types mới giống PhoBot
import streamlit as st

# ==========================================
# 1. CẤU HÌNH TRANG WEB STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Hệ Thống Phân Loại Rác Thông Minh",
    page_icon="♻️",
    layout="wide"
)

# Load biến môi trường từ file .env nếu có
load_dotenv()

# Đồng bộ model thế hệ mới cực nhanh
MODEL_NAME = "gemini-2.5-flash"
USE_MOCK = False 

# Hàm đọc API Key chuẩn bảo mật giống file PhoBot của bạn
def load_api_key():
    try:
        if st.secrets and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
        
    return os.getenv("GEMINI_API_KEY")

api_key = load_api_key()

if not api_key:
    USE_MOCK = True

# Khởi tạo Gemini client theo cú pháp mới giống hệt PhoBot
client = None
if not USE_MOCK and api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        USE_MOCK = True

# Cache để tải Model CNN phân loại rác (Tránh lag web)
@st.cache_resource
def load_my_model():
    return keras.models.load_model('waste_classification_model.keras')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Lỗi không thể tải mô hình CNN (.keras): {e}")

# Hàm xử lý và dự đoán ảnh rác (Hữu cơ O / Tái chế R)
def predict_waste(image):
    img = image.resize((150, 150))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0                 
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction < 0.5:
        confidence = (1 - prediction) * 100
        return "O (Organic - Rác Hữu Cơ)", confidence
    else:
        confidence = prediction * 100
        return "R (Recyclable - Rác Tái Chế)", confidence

# Hàm sinh văn bản giả lập khi mất kết nối mạng
def mock_generate_text(prompt, system_instruction=None):
    return (
        "Xin chào, mình là Trợ lý Rác Thải. Đây là phản hồi mẫu trong chế độ thử nghiệm (Mock mode). "
        "Khi cấu hình API Key chính xác, mình sẽ giải đáp mọi thắc mắc của bạn về môi trường!"
    )

# Hàm gọi Gemini API thật theo chuẩn cấu trúc mới giống PhoBot
def generate_text(prompt, system_instruction=None):
    if USE_MOCK or client is None:
        return mock_generate_text(prompt, system_instruction)

    config = None
    if system_instruction:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        return response.text
    except Exception as error:
        # THAY ĐỔI Ở ĐÂY: In lỗi cụ thể ra màn hình Streamlit thay vì âm thầm bật Mock Mode
        st.error(f"⚠️ Gemini API gặp lỗi thực tế: {error}")
        return "Xin lỗi, hệ thống AI đang bận hoặc gặp sự cố xử lý câu hỏi này. Bạn vui lòng thử lại nhé!"

# Định nghĩa tính cách Chuyên gia Môi trường cho Bot
system_instruction = """
Bạn là một chuyên gia về môi trường và phân loại rác thải tại Việt Nam. Bạn được tích hợp trong một trang web sử dụng model CNN để phân loại rác thải hữu cơ và tái chế.
Nhiệm vụ của bạn:
1. Giải đáp các thắc mắc của người dùng về cách phân loại rác (Hữu cơ, Tái chế, Rác độc hại, vô cơ...).
2. Cung cấp các kiến thức, mẹo tái chế đồ dùng cũ và cập nhật quy định bảo vệ môi trường mới nhất.
3. Hướng dẫn người dùng cách xử lý các loại rác đặc thù (như pin cũ, bóng đèn, rác điện tử).

Quy tắc trả lời:
- Trả lời bằng tiếng Việt, thân thiện, lịch sự và truyền cảm hứng bảo vệ môi trường.
- Câu trả lời nên ngắn gọn, chia đầu dòng rõ ràng, dễ hiểu.
- Không hỏi thêm câu hỏi yes/no nào cho khách hàng vì bạn không thể lưu dữ liệu từ những đoạn chat trước.
"""

def ask_bot(question):
    return generate_text(prompt=question, system_instruction=system_instruction)

# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG STREAMLIT (UI)
# ==========================================
st.title("♻️ Trợ Lý Phân Loại & Tư Vấn Rác Thải Thông Minh")
st.write("Hệ thống tích hợp AI nhận diện thị giác máy tính và Siêu trí tuệ nhân tạo Gemini.")

# Chia giao diện làm 2 cột bằng nhau
col_predict, col_chat = st.columns([1, 1], gap="large")

# --- CỘT TRÁI: CAMERA VÀ UPLOAD ẢNH ---
with col_predict:
    st.header("📸 Nhận Diện Rác Thải")
    
    option = st.radio("Chọn phương thức quét ảnh:", ("Tải ảnh lên (Upload)", "Dùng Camera trực tiếp (Live Cam)"), key="input_method")
    
    uploaded_image = None
    is_valid_file = True  # Biến cờ để kiểm tra file có hợp lệ hay không
    
    if option == "Tải ảnh lên (Upload)":
        # BỎ tham số type=[...] để nút X không bị lỗi giao diện của Streamlit
        file_input = st.file_uploader("Chọn một tấm ảnh rác thải (.jpg, .jpeg, .png)")
        
        if file_input:
            # Tự kiểm tra đuôi file bằng Python
            file_extension = file_input.name.split(".")[-1].lower()
            if file_extension not in ["jpg", "jpeg", "png"]:
                st.error(f"❌ Định dạng file .{file_extension} không được hỗ trợ!")
                st.warning("👉 Vui lòng bấm nút (X) trên thanh file để xóa và chọn lại ảnh đúng (.jpg, .jpeg, .png).")
                is_valid_file = False  # Đánh dấu file lỗi
            else:
                uploaded_image = Image.open(file_input)
    else:
        cam_input = st.camera_input("Đưa rác trước camera của bạn")
        if cam_input:
            uploaded_image = Image.open(cam_input)
            
    # Chỉ chạy AI khi có ảnh và file đó PHẢI HỢP LỆ
    if uploaded_image and is_valid_file:
        st.image(uploaded_image, caption="Ảnh đầu vào", use_container_width=True)
        
        with st.spinner("Đang phân tích hình ảnh..."):
            label, score = predict_waste(uploaded_image)
            
        if "Organic" in label:
            st.success(f"**Kết quả dự đoán:** {label} \n\n**Độ tự tin:** {score:.2f}%")
            st.info("💡 **Gợi ý nhanh:** Rác hữu cơ nên được gom riêng để làm phân bón hữu cơ hoặc xử lý sinh học.")
        else:
            st.warning(f"**Kết quả dự đoán:** {label} \n\n**Độ tự tin:** {score:.2f}%")
            st.info("💡 **Gợi ý nhanh:** Rác tái chế cần được làm sạch, để khô trước khi đưa đến các nhà máy tái chế.")

# --- CỘT PHẢI: CHATBOT GEMINI MÔI TRƯỜNG (ĐÃ FIX KHUNG CUỘN) ---
with col_chat:
    st.header("💬 Trợ Lý Ảo Tư Vấn Môi Trường")
    st.write("Hỏi bất kỳ điều gì về cách phân loại rác, luật môi trường, hoặc mẹo tái chế.")
    
    # Khởi tạo lịch sử chat biến 'messages' y như file PhoBot của bạn
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Mình có thể giúp gì cho bạn trong việc phân loại rác hôm nay?"}
        ]

    # CẢI TIẾN CHÍNH: Tạo một container cố định chiều cao (500px) có thanh cuộn riêng
    # Toàn bộ nội dung tin nhắn chat cũ và mới sẽ được đẩy vào trong box này
    chat_placeholder = st.container(height=500)
    
    with chat_placeholder:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Ô nhập chat đặt ngoài container giúp cố định ở đáy, không bao giờ đè lên tin nhắn mới
    if user_input := st.chat_input("Nhập câu hỏi của bạn (ví dụ: Pin cũ thì vứt ở đâu?)..."):
        
        # 1. Hiển thị ngay tin nhắn của User vào trong khung cuộn
        with chat_placeholder:
            with st.chat_message("user"):
                st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 2. Gọi Bot xử lý câu trả lời và hiển thị tiếp vào trong khung cuộn
        with chat_placeholder:
            with st.chat_message("assistant"):
                with st.spinner("Đang suy nghĩ..."):
                    response = ask_bot(user_input)
                    st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun để Streamlit cập nhật và tự động cuộn xuống tin nhắn cuối cùng
        st.rerun()