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

NUMERICAL_FEATURES = ["lead_time", "adr", "total_of_special_requests", "previous_cancellations", "arrival_date_week_number"]
CATEGORICAL_FEATURES = ["deposit_type", "customer_type"]
TARGET = "is_canceled"

# --- MODELİ ARKA PLANDA HIZLICA EĞİTEN SİHİRLİ FONKSiyon ---
@st.cache_resource
def modeli_ve_ozellikleri_hazirla():
    try:
        df = pd.read_csv("hotel_bookings.csv", low_memory=False)
        df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
        df = df.dropna(subset=[TARGET])
        
        # Sütunların veri setinde var olduğundan emin olalım
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
        
        # Modelin eğitildiği sütun isimlerini ve sıralamasını tam olarak kaydedelim (Hatanın ilacı burası)
        orijinal_X_tasarimi = X_train.copy()
        
        return model_pipeline, orijinal_X_tasarimi
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None

with st.spinner("🤖 MEK Yapay Zeka Modeli Arka Planda Canlı Olarak Eğitiliyor, Lütfen Bekleyin..."):
    model, orijinal_tasarim = modeli_ve_ozellikleri_hazirla()

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

st.title("🏨 MEK: Akıllı Otel Rezervasyon Tahmin Paneli")
st.markdown("Veri madenciliği ve Random Forest algoritması ile rezervasyon risk analizi karar destek sistemi.")
st.divider()

st.sidebar.header("⚙️ Sistem Ayarları")
st.sidebar.info("Bu panel, bulutta dinamik olarak eğitilen ve veri setini kullanan makine öğrenmesi modelini çalıştırmaktadır.")

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
    if model is not None and orijinal_tasarim is not None:
        
        # Pipeline'ın tam olarak beklediği orijinal ham kolon şablonunu oluşturuyoruz
        girdi = pd.DataFrame(columns=orijinal_tasarim.columns)
        
        # Değerleri tam olarak modelin eğitildiği formda yerleştiriyoruz
        girdi.loc[0, "lead_time"] = lead_time
        girdi.loc[0, "adr"] = adr / 35.0  # Euro dengelemesi
        girdi.loc[0, "total_of_special_requests"] = requests
        girdi.loc[0, "previous_cancellations"] = prev_cancellations
        girdi.loc[0, "arrival_date_week_number"] = arrival_week
        girdi.loc[0, "deposit_type"] = deposit
        girdi.loc[0, "customer_type"] = customer_type
        
        # Veri tipini object'ten sayısal değerlere zorlayalım (Pipeline hata vermesin diye)
        for col in NUMERICAL_FEATURES:
            if col in girdi.columns:
                girdi[col] = pd.to_numeric(girdi[col])

        # Tahmin ve Olasılık Hesabı (Artık kolonlar milimetrik olarak eşleşti!)
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
        st.error("Model veya veri şablonu düzgün başlatılamadı.")
