import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi dengan XGBoost",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Aplikasi Prediksi dengan XGBoost")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    try:
        with open('optimal_xgboost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    # Informasi model
    st.sidebar.header("📊 Informasi Model")
    
    # Mendapatkan informasi dari model
    try:
        num_classes = model.n_classes_ if hasattr(model, 'n_classes_') else 4
        st.sidebar.info(f"**Jumlah Kelas:** {num_classes}")
        st.sidebar.info(f"**Tipe Model:** XGBoost Classifier")
        st.sidebar.info(f"**Jumlah Features:** 19")
    except:
        st.sidebar.info("**Model:** XGBoost Classifier (Multiclass)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Petunjuk Penggunaan")
    st.sidebar.markdown("""
    1. Isi semua nilai fitur di bawah
    2. Klik tombol **Prediksi**
    3. Lihat hasil prediksi
    """)
    
    # Main content
    st.markdown("## 📝 Input Fitur")
    st.markdown("Masukkan nilai untuk 19 fitur berikut:")
    
    # Membuat 3 kolom untuk input
    col1, col2, col3 = st.columns(3)
    
    # Inisialisasi list untuk menyimpan nilai input
    input_features = []
    
    # Generate 19 input fields
    with col1:
        st.markdown("**Fitur 1-7**")
        for i in range(1, 8):
            value = st.number_input(
                f"Fitur {i}",
                value=0.0,
                format="%.6f",
                key=f"feature_{i}"
            )
            input_features.append(value)
    
    with col2:
        st.markdown("**Fitur 8-14**")
        for i in range(8, 15):
            value = st.number_input(
                f"Fitur {i}",
                value=0.0,
                format="%.6f",
                key=f"feature_{i}"
            )
            input_features.append(value)
    
    with col3:
        st.markdown("**Fitur 15-19**")
        for i in range(15, 20):
            value = st.number_input(
                f"Fitur {i}",
                value=0.0,
                format="%.6f",
                key=f"feature_{i}"
            )
            input_features.append(value)
    
    st.markdown("---")
    
    # Tombol prediksi
    if st.button("🔮 Prediksi", type="primary", use_container_width=True):
        if len(input_features) == 19:
            # Konversi ke numpy array
            features_array = np.array(input_features).reshape(1, -1)
            
            # Prediksi
            prediction = model.predict(features_array)
            prediction_proba = model.predict_proba(features_array)
            
            # Tampilkan hasil
            st.markdown("## 📈 Hasil Prediksi")
            
            # Layout hasil
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(
                    label="🎯 **Prediksi Kelas**",
                    value=f"Kelas {prediction[0]}"
                )
            
            with res_col2:
                confidence = np.max(prediction_proba[0]) * 100
                st.metric(
                    label="📊 **Tingkat Keyakinan**",
                    value=f"{confidence:.2f}%"
                )
            
            # Tabel probabilitas per kelas
            st.markdown("### Probabilitas per Kelas")
            proba_df = pd.DataFrame({
                'Kelas': [f'Kelas {i}' for i in range(num_classes)],
                'Probabilitas': prediction_proba[0],
                'Persentase': [f"{p*100:.2f}%" for p in prediction_proba[0]]
            })
            
            # Warna berdasarkan probabilitas
            st.dataframe(
                proba_df,
                column_config={
                    "Kelas": st.column_config.TextColumn("Kelas"),
                    "Probabilitas": st.column_config.NumberColumn("Probabilitas", format="%.6f"),
                    "Persentase": st.column_config.TextColumn("Persentase")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Bar chart probabilitas
            st.markdown("### Visualisasi Probabilitas")
            st.bar_chart(proba_df.set_index('Kelas')['Probabilitas'])
            
            # Ringkasan prediksi
            st.success(f"✅ **Kesimpulan:** Data diprediksi termasuk dalam **Kelas {prediction[0]}** dengan keyakinan {confidence:.2f}%")
            
        else:
            st.error("Terjadi kesalahan dalam memproses input")
    
    # Reset button
    if st.button("🔄 Reset", use_container_width=True):
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Aplikasi prediksi menggunakan model XGBoost | Dibuat dengan Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

else:
    st.error("❌ Gagal memuat model. Pastikan file 'optimal_xgboost_model.pkl' berada di direktori yang sama dengan app.py")
    st.info("📁 Struktur folder yang diharapkan:\n```\n├── app.py\n└── optimal_xgboost_model.pkl\n```")