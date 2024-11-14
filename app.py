import streamlit as st
import face_recognition
import os
import cv2
import numpy as np
import psycopg2
import io
from dotenv import load_dotenv
from PIL import Image
from streamlit_option_menu import option_menu

# Load variabel lingkungan dari file .env
load_dotenv()

# Folder untuk menyimpan wajah yang dikenal
KNOWN_FACES_DIR = 'known_faces'
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB

def encode_face(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        return face_encodings[0]  # Mengambil encoding dari wajah pertama yang ditemukan
    else:
        print("Wajah tidak ditemukan di gambar:", image_path)
        return None

# Fungsi untuk menyimpan hasil encoding ke database
def save_encoding_to_db(name, encoding):
    # Konversi encoding ke format list agar bisa disimpan di PostgreSQL
    encoding_list = encoding.tolist() if isinstance(encoding, np.ndarray) else encoding

    try:
        # Koneksi ke database PostgreSQL
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        cur = conn.cursor()
        
        # Query untuk menyisipkan data encoding ke tabel
        insert_query = """
            INSERT INTO users (name, face_encode)
            VALUES (%s, %s);
        """
        cur.execute(insert_query, (name, encoding_list))
        
        # Commit dan tutup koneksi
        conn.commit()
        cur.close()
        conn.close()
        print(f"Encoding untuk {name} berhasil disimpan di database.")
    
    except Exception as e:
        print("Terjadi kesalahan:", e)

def load_encodings_from_db():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        cur = conn.cursor()
        
        cur.execute("SELECT name, face_encode FROM users;")
        rows = cur.fetchall()
        
        # Konversi encoding dari list ke numpy array
        # encodings = [(row[0], np.array(row[1])) for row in rows]
        encodings = [(np.array(row[1])) for row in rows]
        names = [row[0] for row in rows]
        
        cur.close()
        conn.close()
        
        return encodings, names
    
    except Exception as e:
        print("Terjadi kesalahan:", e)
        return [],[]

# Fungsi untuk menambahkan wajah baru
def add_new_face(name, image):
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    # Simpan gambar wajah
    image_path = os.path.join(KNOWN_FACES_DIR, f'{name}.jpg')
    cv2.imwrite(image_path, image)
    image_encoded = encode_face(image_path)
    save_encoding_to_db(name, image_encoded)

# Fungsi untuk mengenali wajah
def recognize_face(uploaded_image):
    # Load known faces
    encodings, names = load_encodings_from_db()

    # Load uploaded image
    uploaded_image = face_recognition.load_image_file(uploaded_image)
    uploaded_face_encodings = face_recognition.face_encodings(uploaded_image)

    face_names = []

    if uploaded_face_encodings:
        for encoding in uploaded_face_encodings:
            matches = face_recognition.compare_faces(encodings, encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]
                face_names.append(name)

    return face_names

# Aplikasi Streamlit
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Tambahkan wajah", "Kenali wajah"]
    )
if selected == "Tambahkan wajah":
    st.title("Aplikasi Pengenalan Wajah")
    # Menambahkan wajah baru
    st.header("Tambahkan Wajah Baru")
    image_get_methode = st.radio(
        "Pilih metode pengambilan gambar anda",
        ["Unggah Gambar", "Ambil dari kamera"],
        captions=[
            "Unggah Gambar",
            "Ambil dari kamera"
        ],
        key='radio_1'
    )

    if image_get_methode == "Unggah Gambar":
        name = st.text_input("Nama:")
        uploaded_file = st.file_uploader("Upload Gambar Wajah", type=["jpg", "jpeg", "png"])

        if st.button("Tambah Wajah"):
            if name and uploaded_file:
                file_size = uploaded_file.size
                if file_size > MAX_FILE_SIZE:
                    st.error("Ukuran file terlalu besar! Harap unggah file di bawah 1 MB.")
                else:
                    face_matched = recognize_face(uploaded_file)
                    if face_matched:
                        st.warning("Wajah sudah terdaftar.")
                    else:
                        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
                        add_new_face(name, image)
                        st.success(f"Wajah untuk {name} berhasil ditambahkan!")
            else:
                st.error("Silakan masukkan nama dan upload gambar.")

    else:
        name = st.text_input("Nama:")
        captured_image = st.camera_input("Ambil gambar dengan kamera")
        if captured_image is not None:
            if name:
                image_check = Image.open(captured_image)
                # Mengonversi gambar PIL menjadi byte stream (sehingga dapat digunakan oleh face_recognition)
                img_byte_arr = io.BytesIO()
                image_check.save(img_byte_arr, format="JPEG")  # Menyimpan gambar sebagai PNG dalam byte stream
                img_byte_arr = img_byte_arr.getvalue() 
                face_matched = recognize_face(io.BytesIO(img_byte_arr))
                if face_matched:
                    st.warning("Wajah sudah terdaftar.")
                else:
                    # Membaca gambar sebagai byte stream
                    image_bytes = captured_image.read()
                    # Konversi byte stream ke array NumPy
                    image_np = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(image_np, 1)
                    add_new_face(name, image)
                    st.success(f"Wajah untuk {name} berhasil ditambahkan!")
            else:
                captured_image = None
                st.error("Silakan masukkan nama.")

if selected == "Kenali wajah":
    st.title("Aplikasi Pengenalan Wajah")
    # Mengenali wajah dari gambar yang diupload
    st.header("Kenali Wajah")
    image_get_methode = st.radio(
        "Pilih metode pengambilan gambar anda",
        ["Unggah Gambar", "Ambil dari kamera"],
        captions=[
            "Unggah Gambar",
            "Ambil dari kamera"
        ],
        key='radio_2'
    )

    if image_get_methode == "Unggah Gambar":
        uploaded_recog_file = st.file_uploader("Upload Gambar untuk Pengenalan", type=["jpg", "jpeg", "png"])
        if st.button("Kenali Wajah"):
            if uploaded_recog_file:
                file_size = uploaded_recog_file.size
                if file_size > MAX_FILE_SIZE:
                    st.error("Ukuran file terlalu besar! Harap unggah file di bawah 1 MB.")
                else:
                    face_names = recognize_face(uploaded_recog_file)
                    if face_names:
                        st.success(f"Wajah yang dikenali: {', '.join(face_names)}")
                    else:
                        st.warning("Tidak ada wajah yang dikenali.")
            else:
                st.error("Silakan upload gambar untuk pengenalan.")

    else:
        captured_image = st.camera_input("Ambil gambar dengan kamera")
        if captured_image is not None:
            image_check = Image.open(captured_image)
            # Mengonversi gambar PIL menjadi byte stream (sehingga dapat digunakan oleh face_recognition)
            img_byte_arr = io.BytesIO()
            image_check.save(img_byte_arr, format="JPEG")  # Menyimpan gambar sebagai PNG dalam byte stream
            img_byte_arr = img_byte_arr.getvalue() 
            face_matched = recognize_face(io.BytesIO(img_byte_arr))
            if face_matched:
                st.success(f"Wajah yang dikenali: {', '.join(face_matched)}")
            else:
                st.warning("Tidak ada wajah yang dikenali.")