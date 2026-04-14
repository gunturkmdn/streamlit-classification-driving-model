import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    with open('model_klasifikasi_pengendara.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Nama fitur berdasarkan model (60 fitur: 30 Acc + 30 Gyro)
FEATURE_NAMES = [
    # Acceleroemeter features (30 features)
    'AccMeanX', 'AccMeanY', 'AccMeanZ',
    'AccCovX', 'AccCovY', 'AccCovZ',
    'AccSkewX', 'AccSkewY', 'AccSkewZ',
    'AccKurtX', 'AccKurtY', 'AccKurtZ',
    'AccSumX', 'AccSumY', 'AccSumZ',
    'AccMinX', 'AccMinY', 'AccMinZ',
    'AccMaxX', 'AccMaxY', 'AccMaxZ',
    'AccVarX', 'AccVarY', 'AccVarZ',
    'AccMedianX', 'AccMedianY', 'AccMedianZ',
    'AccStdX', 'AccStdY', 'AccStdZ',
    # Gyroscope features (30 features)
    'GyroMeanX', 'GyroMeanY', 'GyroMeanZ',
    'GyroCovX', 'GyroCovY', 'GyroCovZ',
    'GyroSkewX', 'GyroSkewY', 'GyroSkewZ',
    'GyroSumX', 'GyroSumY', 'GyroSumZ',
    'GyroKurtX', 'GyroKurtY', 'GyroKurtZ',
    'GyroMinX', 'GyroMinY', 'GyroMinZ',
    'GyroMaxX', 'GyroMaxY', 'GyroMaxZ',
    'GyroVarX', 'GyroVarY', 'GyroVarZ',
    'GyroMedianX', 'GyroMedianY', 'GyroMedianZ',
    'GyroStdX', 'GyroStdY', 'GyroStdZ'
]

# Label kelas (asumsi: 0=Normal, 1=Agressive, 2=Distracted, 3= lainnya)
CLASS_LABELS = {
    0: "🟢 Normal / Safe Driving",
    1: "🟡 Agressive Driving",
    2: "🔴 Distracted Driving",
    3: "⚠️ Other / Unknown"
}

def main():
    st.set_page_config(
        page_title="Driver Behavior Classification",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Driver Behavior Classification System")
    st.markdown("Masukkan data sensor dari accelerometer dan gyroscope untuk mengklasifikasikan perilaku pengendara.")
    
    # Load model
    try:
        model = load_model()
        st.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()
    
    # Sidebar untuk input data
    st.sidebar.header("📊 Input Data Sensor")
    
    # Pilihan input method
    input_method = st.sidebar.radio(
        "Pilih metode input:",
        ["Manual Input", "Upload CSV", "Sample Data"]
    )
    
    input_data = None
    
    if input_method == "Manual Input":
        st.sidebar.subheader("Accelerometer Features")
        acc_col1, acc_col2, acc_col3 = st.sidebar.columns(3)
        
        features = {}
        
        # Accelerometer features
        acc_features = [f for f in FEATURE_NAMES if f.startswith('Acc')]
        for i, feat in enumerate(acc_features):
            col = [acc_col1, acc_col2, acc_col3][i % 3]
            features[feat] = col.number_input(
                feat, 
                value=0.0, 
                format="%.6f",
                key=feat
            )
        
        st.sidebar.subheader("Gyroscope Features")
        gyro_col1, gyro_col2, gyro_col3 = st.sidebar.columns(3)
        
        gyro_features = [f for f in FEATURE_NAMES if f.startswith('Gyro')]
        for i, feat in enumerate(gyro_features):
            col = [gyro_col1, gyro_col2, gyro_col3][i % 3]
            features[feat] = col.number_input(
                feat, 
                value=0.0, 
                format="%.6f",
                key=feat
            )
        
        if st.sidebar.button("🔍 Klasifikasikan", type="primary"):
            input_data = np.array([[features[f] for f in FEATURE_NAMES]])
    
    elif input_method == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload file CSV", 
            type=['csv'],
            help="File CSV harus memiliki 60 kolom sesuai dengan nama fitur"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validasi kolom
                missing_cols = set(FEATURE_NAMES) - set(df.columns)
                if missing_cols:
                    st.sidebar.error(f"Missing columns: {missing_cols}")
                else:
                    input_data = df[FEATURE_NAMES].values
                    st.sidebar.success(f"✅ {len(input_data)} data samples loaded")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
    
    elif input_method == "Sample Data":
        st.sidebar.info("Gunakan sample data untuk testing")
        
        # Sample data normal
        sample_normal = np.zeros(len(FEATURE_NAMES))
        
        # Sample agresif (nilai lebih tinggi)
        sample_aggressive = np.zeros(len(FEATURE_NAMES))
        for i in range(len(FEATURE_NAMES)):
            sample_aggressive[i] = np.random.normal(5, 2)
        
        sample_type = st.sidebar.selectbox(
            "Pilih sample:",
            ["Normal Driver", "Agressive Driver"]
        )
        
        if st.sidebar.button("📋 Gunakan Sample"):
            if sample_type == "Normal Driver":
                input_data = np.array([sample_normal])
            else:
                input_data = np.array([sample_aggressive])
    
    # Main content - Prediction
    if input_data is not None:
        st.subheader("📈 Hasil Klasifikasi")
        
        try:
            # Prediksi
            predictions = model.predict(input_data)
            
            # Untuk model XGBoost dengan multi:softmax
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data)
            else:
                probabilities = None
            
            # Tampilkan hasil
            for i, pred in enumerate(predictions):
                pred_class = int(pred)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        "Predicted Class", 
                        CLASS_LABELS.get(pred_class, f"Class {pred_class}")
                    )
                
                if probabilities is not None and len(probabilities) > i:
                    with col2:
                        st.write("**Probabilitas per kelas:**")
                        prob_df = pd.DataFrame({
                            'Class': [CLASS_LABELS.get(j, f"Class {j}") for j in range(len(probabilities[i]))],
                            'Probability': probabilities[i]
                        })
                        st.dataframe(prob_df, hide_index=True, use_container_width=True)
                
                # Progress bar untuk confidence
                if probabilities is not None and len(probabilities) > i:
                    max_prob = np.max(probabilities[i])
                    st.progress(float(max_prob), text=f"Confidence: {max_prob:.2%}")
            
            # Tampilkan input data (optional)
            with st.expander("📊 Lihat Data Input"):
                input_df = pd.DataFrame(input_data, columns=FEATURE_NAMES)
                st.dataframe(input_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")
            st.info("Pastikan format input data sesuai dengan yang diharapkan model.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>Driver Behavior Classification System | Model: XGBoost Classifier | 4 Classes</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()