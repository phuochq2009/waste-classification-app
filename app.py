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
Bạn là một chuyên gia về môi trường và phân loại rác thải tại Việt Nam. Bạn được tích hợp trong một trang web sử dụng model CNN để phân loại rác thải hữu cơ và tái chế. Có tổng hợp thêm một tab cho người dùng biết được địa điểm thu gom rác đặc thù và độc hại.
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

def ask_gemini_vision(image, cnn_label, cnn_score):
    """Hàm gửi cả ẢNH và KẾT QUẢ CỦA CNN để Gemini chấm điểm, đính chính nếu sai"""
    if USE_MOCK or client is None:
        return (
            "🤖 [Chế độ thử nghiệm]: Gemini đã nhận được ảnh và kết quả của CNN. "
            "Khi có API Key thật, Gemini sẽ đóng vai trò trọng tài để nhận xét kết quả này!"
        )
        
    prompt_referee = f"""
    Bạn là một Chuyên gia Môi trường tối cao đóng vai trò 'Trọng tài công nghệ'. 
    Hệ thống vừa dùng một mô hình học máy CNN để quét bức ảnh này và trả về kết quả là:
    - Dự đoán của CNN: {cnn_label}
    - Độ tự tin: {cnn_score:.2f}%

    Nhiệm vụ của bạn:
    1. Hãy tự nhìn vào bức ảnh này và nhận diện xem vật thể thực tế CHÍNH XÁC là gì.
    2. ĐÁNH GIÁ kết quả của mô hình CNN (Nó đoán ĐÚNG hay SAI?) (thay vì kêu là dự đoán SAI, hãy dùng câu từ nghe cho nó nhẹ nhàng). Nếu nó đoán sai, hãy nhẹ nhàng đính chính lại (nói giảm nói tránh, phân tích tại sao lại sai) loại rác đúng cho người dùng.
    3. Đưa ra hướng xử lý hoặc mẹo tái chế ngắn gọn cho món đồ này.

    Hãy trình bày thật ngắn gọn, mạch lạc bằng tiếng Việt, show hết thông tin ra cho người dùng thấy nhé!
    """
    
    try:
        # Gọi API truyền cả ảnh và prompt
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[image, prompt_referee]
        )
        
        # --- SỬA LỖI NONE Ở ĐÂY ---
        # Kiểm tra xem phản hồi có text hay không, nếu thuộc tính .text bị None, ta quét sâu vào cấu trúc tầng dưới (candidates)
        # --- BỘ LỌC THÔNG MINH KIỂM TRA BỘ LỌC AN TOÀN (SAFETY FILTERS) ---
        if response.text:
            return response.text
            
        # Kiểm tra xem có phải Google trả về kết quả rỗng do chặn nội dung nhạy cảm hay không
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            # Nếu prompt hoặc ảnh bị chặn ngay từ vòng gửi xe
            return "⚠️ [BỘ LỌC AN TOÀN KÍCH HOẠT]: Hình ảnh chứa nội dung nhạy cảm, bạo lực hoặc xác động vật. Google AI Studio đã chủ động chặn đứng (Block) yêu cầu này để đảm bảo an toàn hệ thống."
            
        elif hasattr(response, 'candidates') and response.candidates:
            # Nếu có ứng viên nhưng bị chặn ở tầng trả lời (Finish Reason là SAFETY)
            finish_reason = getattr(response.candidates[0], 'finish_reason', '')
            if finish_reason == 'SAFETY' or finish_reason == 2: # 2 thường là mã số của SAFETY trong SDK
                return "⚠️ [BỘ LỌC AN TOÀN KÍCH HOẠT]: Nội dung hình ảnh vi phạm chính sách bạo lực/ghê rợn (Violence & Gore) của Google. Trọng tài AI từ chối phân tích vật thể này."
            
            try:
                return response.candidates[0].content.parts[0].text
            except:
                pass
                
        # Nếu tất cả đều trống rỗng không rõ nguyên nhân
        return "⚠️ [BỘ LỌC AN TOÀN KÍCH HOẠT]: Server phản hồi một cấu trúc trống. Hình ảnh này đã bị hệ thống kiểm duyệt tự động của Google chặn phân tích (Do nội dung không phù hợp hoặc nhạy cảm)."
            
    except Exception as error:
        return f"⚠️ Trọng tài Gemini gặp sự cố khi nhìn ảnh: {error}"

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

# =========================================================================
# TAB 1: ỨNG DỤNG CHÍNH (NHẬN DIỆN & CHATBOT)
# =========================================================================
with tab_app:

    # Chia giao diện chính làm 2 cột lớn bằng nhau
    col_predict, col_chat = st.columns([1, 1], gap="large")

    # Khởi tạo 2 biến trung gian ở đầu Tab để truyền dữ liệu từ cột Trái sang cột Phải
    label, score = None, None

    # ---------------------------------------------------------------------
    # --- CỘT TRÁI: CAMERA, UPLOAD ẢNH & MÔ HÌNH CNN LOCAL ---
    # ---------------------------------------------------------------------
    with col_predict:
        st.header("📸 Nhận Diện Thị Giác")
        
        option = st.radio("Chọn phương thức quét ảnh:", ("Tải ảnh lên (Upload)", "Dùng Camera trực tiếp (Live Cam)"), key="input_method")
        
        uploaded_image = None
        is_valid_file = True
        
        if option == "Tải ảnh lên (Upload)":
            file_input = st.file_uploader("Chọn một tấm ảnh rác thải (.jpg, .jpeg, .png)")
            
            if file_input:
                file_extension = file_input.name.split(".")[-1].lower()
                if file_extension not in ["jpg", "jpeg", "png"]:
                    st.error(f"❌ Định dạng file .{file_extension} không được hỗ trợ!")
                    is_valid_file = False
                else:
                    uploaded_image = Image.open(file_input)
        else:
            cam_input = st.camera_input("Đưa rác trước camera của bạn")
            if cam_input:
                uploaded_image = Image.open(cam_input)
                
        if uploaded_image and is_valid_file:
            st.image(uploaded_image, caption="Hình ảnh đầu vào hệ thống", use_container_width=True)
            st.markdown("---")
            
            st.subheader("🧠 1. Kết Quả Mô Hình CNN (Local)")
            label, score = predict_waste(uploaded_image)
            
            with st.container(border=True):
                if "Organic" in label:
                    st.success(f"**Kết quả đoán:** {label}\n\n**Độ tự tin máy:** {score:.2f}%")
                else:
                    st.warning(f"**Kết quả đoán:** {label}\n\n**Độ tự tin máy:** {score:.2f}%")

    # ---------------------------------------------------------------------
    # --- CỘT PHẢI: CHATBOT GEMINI MÔI TRƯỜNG & ĐÍNH CHÍNH ẢNH Ở DƯỚI ---
    # ---------------------------------------------------------------------
    with col_chat:
        st.header("💬 Trợ Lý Ảo Tư Vấn Môi Trường")
        st.write("Hỏi bất kỳ điều gì về cách phân loại rác, luật môi trường, hoặc mẹo tái chế.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Xin chào! Mình có thể giúp gì cho bạn trong việc phân loại rác hôm nay?"}
            ]

        # Khung chứa lịch sử chat
        chat_placeholder = st.container(height=380)
        
        with chat_placeholder:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    # --- BONG BÓNG CHAT USER: ÉP CĂN PHẢI, NỀN XANH TÍM CHUYÊN NGHIỆP ---
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                            <div style="background-color: #0078FF; color: white; padding: 10px 14px; 
                                        border-radius: 18px 18px 0px 18px; max-width: 75%; 
                                        text-align: left; box-shadow: 0px 1px 2px rgba(0,0,0,0.15);">
                                {message["content"]}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # --- BONG BÓNG CHAT BOT (ASSISTANT): LỆCH TRÁI, NỀN XÁM NHẠT ---
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="background-color: #F0F2F5; color: #1C1E21; padding: 10px 14px; 
                                        border-radius: 18px 18px 18px 0px; max-width: 75%; 
                                        text-align: left; box-shadow: 0px 1px 2px rgba(0,0,0,0.1);">
                                {message["content"]}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        # Ô nhập câu hỏi chat ghim ở dưới khung chatbot
        if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
            # Thêm tin nhắn user vào bộ nhớ và ép giao diện cập nhật ngay
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Gọi API lấy câu trả lời từ bot
            try:
                response = ask_bot(user_input)
            except Exception as e:
                response = "Xin lỗi, hệ thống AI đang bận hoặc gặp sự cố xử lý câu hỏi này. Bạn vui lòng thử lại nhé!"
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # --- PHẦN ĐÍNH CHÍNH CỦA GEMINI NẰM DƯỚI CÙNG ---
        if uploaded_image and is_valid_file and label is not None:
            st.markdown("---")
            st.subheader("✨ 2. Trọng tài Gemini 2.5 Đính Chính")
            
            with st.spinner("Gemini đang soi ảnh bên trái để check kết quả của CNN..."):
                gemini_vision_result = ask_gemini_vision(uploaded_image, label, score)
            
            with st.container(border=True):
                if "BỘ LỌC AN TOÀN" in gemini_vision_result or "sự cố" in gemini_vision_result or "429" in gemini_vision_result:
                    st.error(gemini_vision_result)
                else:
                    st.markdown(gemini_vision_result)

# ==========================================
# TAB 2: TÍNH NĂNG MỚI - ĐIỂM THU GOM RÁC ĐỘC HẠI
# ==========================================
with tab_locations:
    st.header("📍 Tra Cứu Địa Điểm Thu Gom Rác Đặc Thù & Độc Hại")
    st.write("Pin cũ, bóng đèn hư, rác điện tử chứa nhiều kim loại nặng độc hại, tuyệt đối không vứt chung với rác thông thường. Hãy mang tới các điểm thu gom dưới đây:")
    
    city = st.selectbox("Chọn khu vực của bạn:", ["TP. Hồ Chí Minh", "Hà Nội"])
    
    # Khởi tạo trạng thái focus điểm trong Session State để điều khiển bản đồ
    focus_key = f"selected_point_{city}"
    if focus_key not in st.session_state:
        st.session_state[focus_key] = None

    # CẤU TRÚC DỮ LIỆU ĐÃ ĐỔI SANG ĐƠN VỊ PHƯỜNG CHUẨN (SAU SÁP NHẬP) KÈM THÔNG TIN CHI TIẾT
    if city == "TP. Hồ Chí Minh":
        map_data = [
            {"lat": 10.7963, "lon": 106.7412, "name": "MM Mega Market An Phú", "ward": "Phường An Phú (Quận 2 cũ), TP. Thủ Đức", "addr": "Khu B, KĐT An Phú-An Khánh, TP. Thủ Đức, TP.HCM"},
            {"lat": 10.7932, "lon": 106.7135, "name": "Trạm thu gom UBND Phường 22", "ward": "Phường 22, Quận Bình Thạnh", "addr": "Số 146 Nguyễn Hữu Cảnh, Quận Bình Thạnh, TP.HCM"},
            {"lat": 10.7998, "lon": 106.6802, "name": "Văn phòng Tiếp công dân Phú Nhuận", "ward": "Phường 8, Quận Phú Nhuận", "addr": "Số 159 Nguyễn Văn Trỗi, Quận Phú Nhuận, TP.HCM"},
            {"lat": 10.7850, "lon": 106.6821, "name": "Trung tâm Học tập Cộng đồng Trần Quang Diệu", "ward": "Phường Võ Thị Sáu (sáp nhập Q3)", "addr": "Số 122 Trần Quang Diệu, Quận 3, TP.HCM"}
        ]
    else:
        map_data = [
            {"lat": 21.0245, "lon": 105.8568, "name": "Nhà Văn hóa Phường Tràng Tiền", "ward": "Phường Tràng Tiền, Quận Hoàn Kiếm", "addr": "Số 2 Cổ Tân, Quận Hoàn Kiếm, Hà Nội"},
            {"lat": 21.0115, "lon": 105.8192, "name": "Chi cục Bảo vệ Môi trường Hà Nội", "ward": "Phường Yên Hòa, Quận Cầu Giấy", "addr": "Số 17 Trung Yên 3, Quận Cầu Giấy, Hà Nội"},
            {"lat": 21.0428, "lon": 105.7958, "name": "Trạm tiếp nhận UBND Phường Nghĩa Tân", "ward": "Phường Nghĩa Tân, Quận Cầu Giấy", "addr": "Số 14 Tô Hiệu, Quận Cầu Giấy, Hà Nội"},
            {"lat": 21.0142, "lon": 105.8012, "name": "Điểm thu gom Siêu thị Big C Thăng Long", "ward": "Phường Trung Hòa, Quận Cầu Giấy", "addr": "Số 222 Trần Duy Hưng, Quận Cầu Giấy, Hà Nội"}
        ]

    # Chia giao diện thành 2 cột: Trái hiện danh sách tương tác, Phải hiện Bản đồ lớn
    col_info, col_map = st.columns([1, 1], gap="medium")
    
    # --- CỘT TRÁI: MOVE TOÀN BỘ DANH SÁCH VÀ NÚT BẤM SANG ĐÂY ---
    with col_info:
        st.subheader(f"🏢 Danh Sách Trạm Tiếp Nhận ({city})")
        st.caption("ℹ️ *Hệ thống đã cập nhật tên Phường theo văn bản hành chính mới nhất.*")
        
        # Nút dùng để reset bản đồ hiển thị lại tất cả các điểm ban đầu
        if st.button("🔄 Hiển thị lại toàn bộ các điểm", use_container_width=True):
            st.session_state[focus_key] = None
            st.rerun()

        st.write("")
        
        # Chạy vòng lặp tạo danh sách hộp tương tác
        for idx, pt in enumerate(map_data):
            with st.container(border=True):
                st.markdown(f"**{idx + 1}. {pt['name']}**")
                st.markdown(f"📍 *Đơn vị:* {pt['ward']}")
                st.caption(f"🏠 Địa chỉ: {pt['addr']}")
                
                # Chia 2 nút bấm nằm song song tăm tắp bên dưới mỗi địa điểm
                btn_col1, btn_col2 = st.columns(2)
                
                # NÚT FOCUS THẬT: Khi bấm, lưu điểm được chọn vào session_state để ép bản đồ lọc lại
                if btn_col1.button(f"🎯 Định vị trên Map", key=f"focus_btn_{idx}", use_container_width=True):
                    st.session_state[focus_key] = pt
                    st.toast(f"Đang khóa tiêu điểm vào: {pt['name']}!", icon="🚀")
                    st.rerun()
                
                # Nút mở Tab mới dẫn đường
                google_maps_url = f"https://www.google.com/maps/search/?api=1&query={pt['lat']},{pt['lon']}"
                btn_col2.link_button("🚙 Đường đi (Maps)", google_maps_url, use_container_width=True)

    # --- CỘT PHẢI: CHỈ CHỨA DUY NHẤT BẢN ĐỒ KHỔ LỚN ---
    with col_map:
        st.subheader("🗺️ Bản Đồ Định Vị Vệ Tinh")
        
        # Kiểm tra logic Focus: Nếu người dùng đã nhấn chọn một điểm cụ thể
        if st.session_state[focus_key] is not None:
            selected_pt = st.session_state[focus_key]
            st.info(f"📍 Bản đồ đang thu hẹp tiêu cự vào: **{selected_pt['name']}**")
            
            # Chỉ truyền đúng 1 điểm được chọn vào hàm vẽ để ép st.map tự động Focus căn giữa chính xác vào tọa độ này
            st.map([selected_pt], latitude="lat", longitude="lon", size=60, color="#E65100") 
        else:
            st.caption("ℹ️ *Dùng chuột cuộn để phóng to/thu nhỏ. Các chấm màu xanh lá cây đại diện cho vị trí trạm.*")
            # Nếu chưa chọn gì (hoặc bấm reset), hiển thị toàn bộ danh sách điểm chấm xanh mặc định
            st.map(map_data, latitude="lat", longitude="lon", size=35, color="#2E7D32")

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