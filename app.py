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

# --- TIÊU ĐỀ CHÍNH ---
st.title("♻️ Hệ Thống Quản Lý & Phân Loại Rác Thông Minh")
st.markdown("---")

# --- KHU VỰC THẺ SỐ LIỆU ĐẸP MẮT (METRICS DASHBOARD) ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="Mô Hình Nhận Diện", value="CNN - Keras", delta="Sẵn sàng")
with col_m2:
    st.metric(label="Trí Tuệ Nhân Tạo", value="Gemini 2.5", delta="Đã kết nối")
with col_m3:
    st.metric(label="Độ Phân Giải Chuẩn", value="150x150 px", delta="Đã tối ưu")
with col_m4:
    st.metric(label="Mục Tiêu Hành Tinh", value="Zero Waste", delta="Chung tay")

st.write("")

# --- CẢI TIẾN: PHÂN CHIA TABS CHUYÊN NGHIỆP ---
tab_app, tab_locations, tab_faq = st.tabs([
    "🚀 Ứng Dụng Chính (Nhận Diện & Chat)", 
    "📍 Điểm Thu Gom Rác Độc Hại", 
    "📚 Cẩm Nang Câu Hỏi FAQ"
])

# ==========================================
# TAB 1: ỨNG DỤNG CHÍNH (NHẬN DIỆN & CHATBOT)
# ==========================================
with tab_app:
    # Chia giao diện làm 2 cột bằng nhau
    col_predict, col_chat = st.columns([1, 1], gap="large")

    # --- CỘT TRÁI: CAMERA VÀ UPLOAD ẢNH ---
    with col_predict:
        st.header("📸 Nhận Diện Thị Giác")
        
        option = st.radio("Chọn phương thức quét ảnh:", ("Tải ảnh lên (Upload)", "Dùng Camera trực tiếp (Live Cam)"), key="input_method")
        
        uploaded_image = None
        is_valid_file = True  # Biến cờ để kiểm tra file có hợp lệ hay không
        
        if option == "Tải ảnh lên (Upload)":
            file_input = st.file_uploader("Chọn một tấm ảnh rác thải (.jpg, .jpeg, .png)")
            
            if file_input:
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

    # --- CỘT PHẢI: CHATBOT GEMINI MÔI TRƯỜNG ---
    with col_chat:
        st.header("💬 Trợ Lý Ảo Tư Vấn Môi Trường")
        st.write("Hỏi bất kỳ điều gì về cách phân loại rác, luật môi trường, hoặc mẹo tái chế.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Xin chào! Mình có thể giúp gì cho bạn trong việc phân loại rác hôm nay?"}
            ]

        chat_placeholder = st.container(height=420)
        
        with chat_placeholder:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if user_input := st.chat_input("Nhập câu hỏi của bạn (ví dụ: Pin cũ thì vứt ở đâu?)..."):
            with chat_placeholder:
                with st.chat_message("user"):
                    st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with chat_placeholder:
                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        response = ask_bot(user_input)
                        st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# ==========================================
# TAB 2: TÍNH NĂNG MỚI - ĐIỂM THU GOM RÁC ĐỘC HẠI
# ==========================================
with tab_locations:
    st.header("📍 Tra Cứu Địa Điểm Thu Gom Rác Đặc Thù & Độc Hại")
    st.write("Pin cũ, bóng đèn hư, rác điện tử chứa nhiều kim loại nặng độc hại, tuyệt đối không vứt chung với rác thông thường. Hãy mang tới các điểm thu gom dưới đây:")
    
    city = st.selectbox("Chọn khu vực của bạn:", ["TP. Hồ Chí Minh", "Hà Nội"])
    
    col_info, col_map = st.columns([1, 1], gap="medium")
    
    with col_info:
        if city == "TP. Hồ Chí Minh":
            st.subheader("🏢 Các Điểm Tiếp Nhận Tại TP.HCM")
            st.markdown("""
            *   **Quận 1:** MM Mega Market An Phú (Khu B, KĐT An Phú-An Khánh).
            *   **Quận 3:** Trung tâm Học tập Cộng đồng (Số 122 Trần Quang Diệu).
            *   **Quận Bình Thạnh:** Ủy ban Nhân dân Phường 22 (Số 146 Nguyễn Hữu Cảnh).
            *   **Quận Phú Nhuận:** Văn phòng Tiếp công dân (Số 159 Nguyễn Văn Trỗi).
            *   *Mẹo:* Gom pin vào chai nhựa sạch trước khi mang đi nộp để đảm bảo an toàn vận chuyển.
            """)
        else:
            st.subheader("🏢 Các Điểm Tiếp Nhận Tại Hà Nội")
            st.markdown("""
            *   **Quận Hoàn Kiếm:** Nhà Văn hóa Phường Tràng Tiền (Số 2 Cổ Tân).
            *   **Quận Ba Đình:** Chi cục Bảo vệ Môi trường Hà Nội (Số 17 Trung Yên 3).
            *   **Quận Cầu Giấy:** UBND Phường Nghĩa Tân (Số 14 Tô Hiệu).
            *   **Quận Thanh Xuân:** Điểm thu gom tại Siêu thị Big C Thăng Long (Số 222 Trần Duy Hưng).
            *   *Mẹo:* Các điểm thu gom rác điện tử thường hoạt động trong giờ hành chính các ngày trong tuần.
            """)
            
    with col_map:
        st.subheader("🗺️ Bản Đồ Trực Quan Quốc Gia")
        # Nhúng bản đồ Google Maps của dự án Việt Nam Tái Chế để giao diện trông siêu thực tế
        map_html = """
        <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3919.5201944607753!2d106.69916297583796!3d10.771415559283738!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x31752f40a3b090b5%3A0x8878b3fbdf68153c!2zVUJORCBRdeG6rW4gMQ!5e0!3m2!1svi!2svn!4v1700000000000!5m2!1svi!2svn" 
        width="100%" height="250" style="border:0; border-radius:10px;" allowfullscreen="" loading="lazy"></iframe>
        """
        st.components.v1.html(map_html, height=260)

# ==========================================
# TAB 3: TÍNH NĂNG MỚI - CẨM NANG CÂU HỎI FAQ
# ==========================================
with tab_faq:
    st.header("📚 Các Câu Hỏi Thường Gặp Về Phân Loại Rác")
    st.write("Nhấp vào từng câu hỏi dưới đây để mở xem câu trả lời nhanh từ các chuyên gia môi trường:")
    
    with st.expander("❓ Hộp xốp, ly nhựa dính dầu mỡ ăn xong có bỏ vào thùng rác tái chế không?"):
        st.write("👉 **Không.** Mặc dù chúng làm bằng nhựa/xốp nhưng một khi đã dính dầu mỡ nặng, chi phí công nghệ để sục rửa sạch thu hồi hạt nhựa đắt hơn rất nhiều so với sản xuất đồ mới. Vì vậy, chúng được gom vào loại **Rác còn lại (Không tái chế)**.")
        
    with st.expander("❓ Tại sao không được bỏ pin cũ và đồ điện tử vào thùng rác gia đình thông thường?"):
        st.write("👉 **Vì nhiễm độc nguồn nước và đất.** Pin chứa chì, thủy ngân, cadmium. Khi bị chôn lấp vỡ vỏ, hóa chất độc hại sẽ ngấm xuống mạch nước ngầm hoặc bốc lên không khí độc hại khi bị đốt cháy thủ công. Bắt buộc phải gom riêng mang đến điểm xử lý rác công nghệ cao.")
        
    with st.expander("❓ Vỏ sò, vỏ ốc, xương gà vịt heo bò là rác hữu cơ hay rác còn lại?"):
        st.write("👉 **Xương nhỏ (gà, cá), vỏ trứng, rau củ $\rightarrow$ Rác hữu cơ.** Tuy nhiên, các loại **xương ống bò/heo lớn, vỏ ngao, vỏ sò cứng** rất khó để các máy xay rác hữu cơ nghiền nhỏ làm phân bón. Do đó tại nhiều địa phương, chúng được xếp vào nhóm **Rác còn lại**.")