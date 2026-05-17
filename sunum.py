import streamlit as st
import pandas as pd
import joblib

# Sayfa yapılandırması (Tarayıcı sekmesindeki başlık ve ikon)
st.set_page_config(page_title="Otel İptal Analizi", page_icon="🏨", layout="wide")

# 1. Modeli ve Sütunları Yükle
@st.cache_resource
def model_yukle():
    return joblib.load('otel_tahmin_modeli.pkl')

paket = model_yukle()
model = paket["model"]
sutunlar = paket["columns"]

# --- ARAYÜZ TASARIMI ---
st.title("🏨 Akıllı Otel Rezervasyon Tahmin Paneli")
st.markdown("Veri madenciliği ve Random Forest algoritması ile rezervasyon risk analizi.")
st.divider()

# Yan Menü (Sidebar) - Genel Ayarlar
st.sidebar.header("⚙️ Sistem Ayarları")
st.sidebar.info("Bu panel, 119.000 kayıtlık veri setiyle eğitilmiş makine öğrenmesi modelini kullanmaktadır.")

# Ana Ekranı Sütunlara Bölelim
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Zamanlama")
    lead_time = st.number_input("Rezervasyon Kaç Gün Önce Yapıldı?", min_value=0, max_value=730, value=50)
    arrival_week = st.slider("Varış Haftası (Yılın kaçıncı haftası?)", 1, 53, 25)
    stay_nights = st.number_input("Toplam Konaklama Gecesi", min_value=1, value=3)

with col2:
    st.subheader("💰 Finansal Detaylar")
    adr = st.number_input("Günlük Oda Fiyatı (€)", min_value=0.0, value=120.0)
    deposit = st.selectbox("Depozito Tipi", ["No Deposit", "Non Refund", "Refundable"])
    customer_type = st.selectbox("Müşteri Tipi", ["Transient", "Contract", "Group"])

with col3:
    st.subheader("👤 Müşteri Profili")
    requests = st.slider("Özel İstek Sayısı", 0, 5, 0)
    prev_cancellations = st.number_input("Geçmiş İptal Sayısı", min_value=0, value=0)
    parking = st.checkbox("Otopark Alanı İstiyor mu?")

st.divider()

# --- TAHMİN MANTIĞI ---
if st.button("🚀 RİSK ANALİZİNİ ÇALIŞTIR", use_container_width=True):
    # 1. Boş bir veri çerçevesi oluştur
    girdi = pd.DataFrame(0, index=[0], columns=sutunlar)
    
    # 2. Temel değerleri ata
    girdi["lead_time"] = lead_time
    girdi["adr"] = adr
    girdi["total_of_special_requests"] = requests
    girdi["previous_cancellations"] = prev_cancellations
    girdi["arrival_date_week_number"] = arrival_week
    girdi["stays_in_week_nights"] = stay_nights
    
    # 3. Kategorik (Seçmeli) verileri modele uygun hale getir
    if f"deposit_type_{deposit}" in sutunlar:
        girdi[f"deposit_type_{deposit}"] = 1
    if f"customer_type_{customer_type}" in sutunlar:
        girdi[f"customer_type_{customer_type}"] = 1
    if parking and "required_car_parking_spaces" in sutunlar:
        girdi["required_car_parking_spaces"] = 1

    # 4. Tahmin ve Olasılık Hesabı
    tahmin = model.predict(girdi)[0]
    olasılık = model.predict_proba(girdi) # [İptal Etmeme %, İptal Etme %]

    # --- SONUÇ EKRANI ---
    st.subheader("📊 Analiz Sonucu")
    
    if tahmin == 1:
        st.error(f"### YÜKSEK RİSK: Bu rezervasyon muhtemelen İPTAL EDİLECEK.")
        st.metric("İptal Olasılığı", f"%{olasılık[0][1]*100:.1f}")
        st.warning("Öneri: Müşteriyle iletişime geçin veya depozito onayını kontrol edin.")
    else:
        st.success(f"### DÜŞÜK RİSK: Bu rezervasyon güvenli görünüyor.")
        st.metric("Geliş Olasılığı", f"%{olasılık[0][0]*100:.1f}")
        st.info("Öneri: Standart operasyon sürecine devam edebilirsiniz.")