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
st.set_page_config(page_title="MEK Otel İptal Analizi", page_icon="🏨", layout="wide")

# --- MODELİ ARKA PLANDA HIZLICA EĞİTEN SİHİRLİ FONKSİYON ---
@st.cache_resource
def modeli_ve_sutunlari_hazirla():
    try:
        # Veri setini yükle
        df = pd.read_csv("hotel_bookings.csv", low_memory=False)
        df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
        df = df.dropna(subset=["is_canceled"])
        
        NUMERICAL_FEATURES = ["lead_time", "adr", "total_of_special_requests", "previous_cancellations", "arrival_date_week_number"]
        CATEGORICAL_FEATURES = ["deposit_type", "customer_type"]
        TARGET = "is_canceled"
        
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        y = df[TARGET]
        
        # Hızlı çalışması için dengeli bir alt küme alalım (Hızı ve RAM'i optimize etmek için)
        X_train, _, y_train, _ = train_test_split(X, y, train_size=0.30, random_state=42, stratify=y)
        
        # İşlemciler (Preprocessor)
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipe, NUMERICAL_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES)
        ])
        
        # Canlı Eğitim Hattı (Boyutu şişirmemek ve hızlı açılmak için ağaç sayısını 50 yaptık)
        model_pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
        ])
        
        model_pipeline.fit(X_train, y_train)
        
        # Modele giren kolon isimlerini yapılandıralım
        preprocessor.fit(X_train)
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['ohe']
        encoded_cat_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        butun_sutunlar = NUMERICAL_FEATURES + encoded_cat_cols + ["required_car_parking_spaces"]
        
        return model_pipeline, butun_sutunlar
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, []

# Canlı modeli yükle (Streamlit bunu bir kez yapıp hafızada tutacak)
with st.spinner("🤖 MEK Yapay Zeka Modeli Arka Planda Canlı Olarak Eğitiliyor, Lütfen Bekleyin..."):
    model, sutunlar = modeli_ve_sutunlari_hazirla()

# --- DİL VE VERİ EŞLEŞTİRME SÖZLÜKLERİ ---
depozito_sozlugu = {
    "Depozito Yok (No Deposit)": "No Deposit",
    "İade Edilemez (Non Refund)": "Non Refund",
    "İade Edilebilir (Refundable)": "Refundable"
}

musteri_sozlugu = {
    "Bireysel Müşteri (Transient)": "Transient",
    "Sözleşmeli / Kurumsal (Contract)": "Contract",
    "Grup Rezervasyonu (Group)": "Group"
}

# --- ARAYÜZ TASARIMI ---
st.title("🏨 MEK: Akıllı Otel Rezervasyon Tahmin Paneli")
st.markdown("Veri madenciliği ve Random Forest algoritması ile rezervasyon risk analizi karar destek sistemi.")
st.divider()

st.sidebar.header("⚙️ Sistem Ayarları")
st.sidebar.info("Bu panel, GitHub dosya boyutu limitlerini aşmak için bulutta dinamik olarak eğitilen ve 119.000 kayıtlık veri setini kullanan makine öğrenmesi modelini çalıştırmaktadır.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Zamanlama")
    lead_time = st.number_input("Rezervasyon Kaç Gün Önce Yapıldı? (Lead Time)", min_value=0, max_value=730, value=50)
    arrival_week = st.slider("Varış Haftası (Yılın kaçıncı haftası?)", 1, 53, 25)
    stay_nights = st.number_input("Toplam Konaklama Gecesi", min_value=1, value=3)

with col2:
    st.subheader("💰 Finansal Detaylar")
    adr = st.number_input("Günlük Ortalama Oda Fiyatı (TL - ADR)", min_value=0.0, value=4000.0)
    secilen_depozito_tr = st.selectbox("Depozito Tipi", list(depozito_sozlugu.keys()))
    deposit = depozito_sozlugu[secilen_depozito_tr]
    
    secilen_musteri_tr = st.selectbox("Müşteri Tipi", list(musteri_sozlugu.keys()))
    customer_type = musteri_sozlugu[secilen_musteri_tr]

with col3:
    st.subheader("👤 Müşteri Profili")
    requests = st.slider("Özel İstek Sayısı", 0, 5, 0)
    prev_cancellations = st.number_input("Geçmiş İptal Edilen Rezervasyon Sayısı", min_value=0, value=0)
    parking = st.checkbox("Otopark Alanı Talep Ediyor mu?")

st.divider()

if st.button("🚀 RİSK ANALİZİNİ ÇALIŞTIR", use_container_width=True):
    if model is not None:
        girdi = pd.DataFrame(0, index=[0], columns=sutunlar)
        
        girdi["lead_time"] = lead_time
        girdi["adr"] = adr / 35.0  # Euro veri tabanına göre dengeleme
        girdi["total_of_special_requests"] = requests
        girdi["previous_cancellations"] = prev_cancellations
        girdi["arrival_date_week_number"] = arrival_week
        girdi["stays_in_week_nights"] = stay_nights
        
        if f"deposit_type_{deposit}" in sutunlar:
            girdi[f"deposit_type_{deposit}"] = 1
        if f"customer_type_{customer_type}" in sutunlar:
            girdi[f"customer_type_{customer_type}"] = 1
        if parking and "required_car_parking_spaces" in sutunlar:
            girdi["required_car_parking_spaces"] = 1

        girdi = girdi.fillna(0)
        
        # Tahmin ve Olasılık Hesabı
        tahmin = model.predict(girdi)[0]
        olasılık = model.predict_proba(girdi)

        st.subheader("📊 Analiz Sonucu")
        
        if tahmin == 1:
            st.error(f"### YÜKSEK RİSK: Bu rezervasyon muhtemelen İPTAL EDİLECEK.")
            st.metric("İptal Etme Riski Olasılığı", f"%{olasılık[0][1]*100:.1f}")
            st.warning("🚨 Öneri: Müşteriyle proaktif olarak iletişime geçin, teyit alın veya depozito durumunu güncelleyin.")
        else:
            st.success(f"### DÜŞÜK RİSK: Bu rezervasyon güvenli görünüyor. Müşteri otele giriş yapacaktır.")
            st.metric("Otele Geliş (Kesinlik) Olasılığı", f"%{olasılık[0][0]*100:.1f}")
            st.info("✅ Öneri: Standart operasyon sürecine ve oda hazırlığına devam edebilirsiniz.")
    else:
        st.error("Model eğitilemediği için tahmin yapılamıyor.")