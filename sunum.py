import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# Sayfa yapılandırması
st.set_page_config(page_title="Otel Rezervasyon İptal Tahmini", layout="wide")

NUMERICAL_FEATURES = ["lead_time", "adr", "total_of_special_requests", "previous_cancellations", "arrival_date_week_number"]
CATEGORICAL_FEATURES = ["deposit_type", "customer_type"]
TARGET = "is_canceled"

# Model eğitim fonksiyonu
@st.cache_resource
def modeli_ve_ozellikleri_hazirla():
    try:
        df = pd.read_csv("hotel_bookings.csv", low_memory=False)
        df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
        df = df.dropna(subset=[TARGET])
        
        mevcut_num = [col for col in NUMERICAL_FEATURES if col in df.columns]
        mevcut_cat = [col for col in CATEGORICAL_FEATURES if col in df.columns]
        
        X = df[mevcut_num + mevcut_cat]
        y = df[TARGET]
        
        X_train, _, y_train, _ = train_test_split(X, y, train_size=0.30, random_state=42, stratify=y)
        
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipe, mevcut_num),
            ("cat", cat_pipe, mevcut_cat)
        ])
        
        model_pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
        ])
        
        model_pipeline.fit(X_train, y_train)
        
        orijinal_X_tasarimi = X_train.copy()
        
        return model_pipeline, orijinal_X_tasarimi
    except Exception as e:
        st.error(f"Model yuklenirken hata olustu: {e}")
        return None, None

with st.spinner("Model Arka Planda Canli Olarak Egitiliyor, Lutfen Bekleyin..."):
    model, orijinal_tasarim = modeli_ve_ozellikleri_hazirla()

depozito_sozlugu = {
    "Depozito Yok (No Deposit)": "No Deposit",
    "Iade Edilemez (Non Refund)": "Non Refund",
    "Iade Edilebilir (Refundable)": "Refundable"
}

musteri_sozlugu = {
    "Bireysel Musteri (Transient)": "Transient",
    "Sozlesmeli / Kurumsal (Contract)": "Contract",
    "Grup Rezervasyonu (Group)": "Group"
}

st.title("Otel Rezervasyon İptal Tahmini")
st.markdown("Random Forest Algoritmasi ile Gelistirilmis Akilli Karar Destek Sistemi")
st.divider()

st.sidebar.header("Sistem Ayarlari")
st.sidebar.info("Bu panel, bulutta dinamik olarak egitilen ve veri setini kullanan makine ogrenmesi modelini calistirmaktadir.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Zamanlama")
    lead_time = st.number_input("Rezervasyon Kac Gun Once Yapildi? (Lead Time)", min_value=0, max_value=730, value=50)
    arrival_week = st.slider("Varis Haftasi (Yilin kacinci haftasi?)", 1, 53, 25)
    stay_nights = st.number_input("Toplam Konaklama Gecesi", min_value=1, value=3)

with col2:
    st.subheader("Finansal Detaylar")
    adr = st.number_input("Gunluk Ortalama Oda Fiyati (TL - ADR)", min_value=0.0, value=4000.0)
    secilen_depozito_tr = st.selectbox("Depozito Tipi", list(depozito_sozlugu.keys()))
    deposit = depozito_sozlugu[secilen_depozito_tr]
    
    secilen_musteri_tr = st.selectbox("Musteri Tipi", list(musteri_sozlugu.keys()))
    customer_type = musteri_sozlugu[secilen_musteri_tr]

with col3:
    st.subheader("Musteri Profili")
    requests = st.slider("Ozel Istek Sayisi", 0, 5, 0)
    prev_cancellations = st.number_input("Gecmis Iptal Edilen Rezervasyon Sayisi", min_value=0, value=0)
    parking = st.checkbox("Otopark Alani Talep Ediyor mu?")

st.divider()

if st.button("RISK ANALIZINI CALISTIR", use_container_width=True):
    if model is not None and orijinal_tasarim is not None:
        
        girdi = pd.DataFrame(columns=orijinal_tasarim.columns)
        
        girdi.loc[0, "lead_time"] = lead_time
        girdi.loc[0, "adr"] = adr / 35.0  
        girdi.loc[0, "total_of_special_requests"] = requests
        girdi.loc[0, "previous_cancellations"] = prev_cancellations
        girdi.loc[0, "arrival_date_week_number"] = arrival_week
        girdi.loc[0, "deposit_type"] = deposit
        girdi.loc[0, "customer_type"] = customer_type
        
        for col in NUMERICAL_FEATURES:
            if col in girdi.columns:
                girdi[col] = pd.to_numeric(girdi[col])

        tahmin = model.predict(girdi)[0]
        olasılık = model.predict_proba(girdi)

        st.subheader("Analiz Sonucu")
        
        if tahmin == 1:
            st.error("YUKSEK RISK: Bu rezervasyon muhtemelen IPTAL EDILECEK.")
            st.metric("Iptal Etme Riski Olasiligi", f"%{olasılık[0][1]*100:.1f}")
            st.warning("Oneri: Musteriyle proaktif olarak iletisime gecin, teyit alin veya depozito durumunu guncelleyin.")
        else:
            st.success("DUSUK RISK: Bu rezervasyon guvenli gorunuyor. Musteri otele giris yapacaktir.")
            st.metric("Otele Gelis (Kesinlik) Olasiligi", f"%{olasılık[0][0]*100:.1f}")
            st.info("Oneri: Standart operasyon surecine ve oda hazirligina devam edebilirsiniz.")
    else:
        st.error("Model veya veri sablonu duzgun baslatilamadi.")
