import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
# Import cần thiết cho việc cấu hình System Instruction trong Chat Session
from google.genai.types import GenerativeConfig 

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(col, errors='coerce').fillna(0) # Sửa lỗi: Cần dùng df[col]
    
    # Sửa lỗi logic: Chuyển đổi cột thành số trước khi tính toán
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        # Nếu không tìm thấy, cố gắng tìm các tên gọi khác (chẳng hạn V. TỔNG TÀI SẢN)
        tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TÀI SẢN$', case=False, na=False, regex=True)]
        if tong_tai_san_row.empty:
             raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN' hoặc các chỉ tiêu tương đương.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý chia cho 0 thủ công cho giá trị đơn lẻ
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Tài chính (Giữ nguyên) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# **********************************************
# --- KHU VỰC THÊM CHỨC NĂNG CHAT GEMINI (ĐÃ SỬA LỖI) ---
# **********************************************

# Hàm khởi tạo và lấy Chat Session (Đã sửa lỗi tham số system_instruction)
def get_chat_session():
    """Khởi tạo hoặc trả về Chat Session hiện tại."""
    api_key = st.secrets.get("GEMINI_API_KEY") 
    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API để khởi tạo Chatbot. Vui lòng kiểm tra Streamlit Secrets.")
        return None
        
    # Tạo client và session
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Thiết lập lịch sử chat trong session state
        if "chat_session" not in st.session_state:
            # 1. Định nghĩa System Instruction
            system_instruction = "Bạn là một trợ lý phân tích tài chính thân thiện, chuyên nghiệp, chỉ trả lời các câu hỏi liên quan đến tài chính, kế toán hoặc các chủ đề kinh tế chung. Luôn trả lời bằng Tiếng Việt."
            
            # 2. Tạo đối tượng GenerativeConfig
            config = GenerativeConfig(
                system_instruction=system_instruction
            )
            
            # 3. Truyền config vào client.chats.create()
            st.session_state.chat_session = client.chats.create(
                model=model_name,
                config=config  # SỬA LỖI: Truyền thông qua tham số config
            )
            # Khởi tạo lịch sử hiển thị
            st.session_state.messages = []
            
        return st.session_state.chat_session
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo Gemini Client: {e}")
        return None

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Khởi tạo các biến để tránh lỗi UnboundLocalError
df_processed = None
data_for_ai = None
thanh_toan_hien_hanh_N = "N/A"
thanh_toan_hien_hanh_N_1 = "N/A"

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n_row = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]
                tsnh_n = tsnh_n_row['Năm sau'].iloc[0] if not tsnh_n_row.empty else 0
                tsnh_n_1 = tsnh_n_row['Năm trước'].iloc[0] if not tsnh_n_row.empty else 0

                # Lấy Nợ ngắn hạn
                no_ngan_han_row = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]
                no_ngan_han_N = no_ngan_han_row['Năm sau'].iloc[0] if not no_ngan_han_row.empty else 0
                no_ngan_han_N_1 = no_ngan_han_row['Năm trước'].iloc[0] if not no_ngan_han_row.empty else 0

                # Tính toán, xử lý chia cho 0
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')
                
                col1, col2 = st.columns(2)
                with col1:
                    value_n_1 = f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 != float('inf') else '∞'
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=value_n_1
                    )
                with col2:
                    value_n = f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N != float('inf') else '∞'
                    delta_value = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}" if (thanh_toan_hien_hanh_N != float('inf') and thanh_toan_hien_hanh_N_1 != float('inf')) else None
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=value_n,
                        delta=delta_value
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                 thanh_toan_hien_hanh_N = "N/A" 
                 thanh_toan_hien_hanh_N_1 = "N/A"
            except ZeroDivisionError:
                 st.warning("Nợ ngắn hạn bằng 0, chỉ số thanh toán hiện hành là vô cùng (∞).")
                 
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)].empty else 'N/A', 
                    f"{thanh_toan_hien_hanh_N}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                     st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# **********************************************
# --- KHUNG CHAT HỎI ĐÁP VỚI GEMINI (ĐÃ TÍCH HỢP VÀ SỬA LỖI) ---
# **********************************************

# Chỉ hiển thị khung chat khi đã tải file và xử lý xong
if uploaded_file is not None and df_processed is not None:
    st.markdown("---")
    st.subheader("6. Hỏi đáp chuyên sâu với Gemini 🤖")
    
    # 1. Khởi tạo Chat Session
    chat = get_chat_session()
    
    if chat:
        # 2. Hiển thị lịch sử tin nhắn
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 3. Xử lý đầu vào từ người dùng
        if prompt := st.chat_input("Hỏi Gemini về các vấn đề tài chính, ví dụ: 'Chỉ số thanh toán hiện hành bao nhiêu là tốt?'"):
            # Lưu tin nhắn người dùng vào lịch sử
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Gửi câu hỏi đến Gemini và nhận phản hồi
            with st.chat_message("assistant"):
                with st.spinner("Gemini đang trả lời..."):
                    try:
                        # Gửi nội dung tin nhắn và nhận phản hồi (stream để hiển thị mượt mà hơn)
                        response = chat.send_message(prompt, stream=True)
                        response_text = st.write_stream(response)
                        
                        # Lưu phản hồi vào lịch sử
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

                    except APIError as e:
                        st.error(f"Lỗi API: {e}. Vui lòng kiểm tra lại GEMINI_API_KEY.")
                    except Exception as e:
                        st.error(f"Đã xảy ra lỗi không xác định: {e}")

# **********************************************
# --- KẾT THÚC CHỨC NĂNG CHAT GEMINI ---
# **********************************************
