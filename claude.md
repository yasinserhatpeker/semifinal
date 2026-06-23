# 🧬 TEKNOFEST 2026 - Sağlıkta Yapay Zeka Yarışması
## 🏆 CFTR Veri Seti Şampiyonluk Yol Haritası & Strateji Kılavuzu (`claude.md`)

Bu doküman, **CFTR** genetik varyant patojenite tahmini veri kümesi üzerinde yapılan derinlemesine analizler, karşılaşılan metodolojik tuzaklar ve PDR (Proje Detay Raporu) aşamasında tam puan almamızı sağlayacak mimari kararların bir derlemesidir.

---

## 🔍 1. Veri Kümesi Röntgeni & Mevcut Durum Analizi

* **Boyut ve Geometri:** 111 satır (örnek) ve 353 sütun (özellik). Biyoinformatikteki tipik **"Küçük N, Büyük P"** problemi. Yüksek overfitting (aşırı öğrenme) riski barındırıyor.
* **Sınıf Dengesizliği (Class Imbalance):** `Label` sütununda 90 tane `1` (Patojenik) ve sadece 21 tane `0` (Benign/Zararsız) sınıfı bulunuyor (%81'e %19). Sabit 0.50 eşik değeriyle eğitilen modeller ezbere `1` demeye meyillidir.
* **Eksik Veri (NaN) Yapısı:** `AL_` (Allel Frekansı) sütunlarında çok ciddi miktarda eksik veri mevcut. Biyolojik bağlamda bu durum, varyantın popülasyonda hiç kaydedilmediğini (yani aşırı nadir ve patojenite potansiyeli yüksek olduğunu) gösterir.

---

## 🛠️ 2. Alınan Doğru Kararlar & Öznitelik Mühendisliği (Feature Engineering)

Modelin salt istatistiksel karmaşada kaybolmasını önlemek ve biyolojik anlam kazandırmak adına şu kararlar alınmış ve kodlanmıştır:

1.  **Amino Asit Değişim Matrisleri (`AA_`):** `AA_1` (Referans) ve `AA_2` (Alternatif) amino asitleri harf bazlı bırakılmamış; evrimsel ikame olasılıklarını gösteren **BLOSUM62** ve **PAM250** matrisleri koda gömülerek satır bazlı skorlara dönüştürülmüştür. Bu özellikler ilk testlerde en yüksek katkıyı sağlayan ilk 8 özellik arasına girmiştir.
2.  **Fizikokimyasal Mesafeler:** Amino asitlerin Kyte-Doolittle ölçeğine göre hidrofobiklik farkları hesaplanarak protein yapısındaki katlanma hasarı simüle edilmiştir.
3.  **Nadir Varyant Sinyali (`FEATURE_Missing_Count`):** Satır bazındaki toplam `NaN` sayısı saydırılarak, veri tabanlarında bulunmayan "aşırı nadir varyantlar" için güçlü bir yapay özellik üretilmiştir.
4.  **Evrimsel Korunmuşluk Özetleri (`EK_`):** `EK_1`'den `EK_9`'u kadar olan tüm ham korunmuşluk skorları korunmuş, ekstra olarak satır bazlı `mean`, `max` ve `std` (standart sapma) değerleri modele sunulmuştur. (`EK_9` modeli %9.37 önem derecesiyle tek başına domine etmiştir).
5.  **Frekans İstatistikleri (`AL_`):** 334 adet `AL_` sütununun satır bazlı `mean`, `max` ve `std` dağılımları çıkarılmıştır.

---

## 📐 3. Doğrulama Stratejisi & Overfitting Önlemleri

* **Stratified K-Fold Cross Validation:** Küçük veri setlerinde `train_test_split` tuzağına düşülmemiştir. Veri seti rastgele 5 kata (Fold) bölünmüş ve her katın ana veri setindeki %81-%19 dengesini koruması sağlanmıştır (Stratified). Modelin test aşamasında aldığı **%94.44 OOF F1-Score** ve **%85.79 ROC-AUC Skoru** modelin ezberlemediğinin (overfitting yapmadığının) matematiksel kanıtıdır.
* **Ayar Parametreleri (Muhafazakar Hiperparametreler):** Modelin veriyi ezberlememesi için ağaç derinliği (`depth=4`) sığ tutulmuş ve yapraklara kısıtlama getiren `l2_leaf_reg=5` düzenlileştirmesi (regularization) eklenmiştir.
* **Aykırı Değer (Outlier) Yönetimi:** Genomik verideki uç değerleri silmek biyolojik bilgiyi yok edeceğinden veriler kırpılmamıştır. Karar ağaçları mimarisine sahip **CatBoost** seçilerek aykırı değerlere karşı doğal bir dayanıklılık (robustness) sağlanmıştır.

---

## 🎯 4. İlerleyen Aşamalar İçin Geliştirme Planı (Ensemble & Optuna)

Modeli birinciliğe taşımak adına kod mimarisine eklenecek ve PDR raporunda savunulacak sonraki faz adımları:

1.  **Grantham Distance (Grantham Mesafesi):** İki amino asit arasındaki kimyasal formül, atomik hacim ve polarite farkını tek bir formülle birleştiren klasik biyoinformatik metriği koda gömülecektir.
2.  **AL_ Sütun Bloklaması (Feature Binning):** 334 adet `AL_` sütunu, biyolojik popülasyon kökenlerine göre (Avrupa, Asya, Afrika vb.) 4'erli veya 5'erli lokal alt gruplara bölünecek ve her grubun kendi iç istatistikleri çıkarılarak model üzerindeki boyut yükü hafifletilecektir.
3.  **Bayesian Optimizasyon (Optuna):** Parametrelerin manuel seçimi yerine, olasılıksal yaklaşımla en optimum kombinasyonu bulan Optuna entegrasyonu tamamlanacaktır.
4.  **Mükemmel Topluluk (Ensemble - Soft Voting):** Aynı K-Fold katlarında eğitilmiş **CatBoost + LightGBM + XGBoost** modellerinin ürettiği olasılık tahminlerinin (`predict_proba`) ağırlıklı ortalaması alınarak nihai jüri test seti tahmini yapılacaktır.

---

## 📝 5. PDR Raporu İçin Jüri Savunma Notları

* **Soru: Azınlık sınıfı (Benign) için SMOTE gibi oversampling neden yapılmadı?**
    * *Cevap:* 21 satırlık aşırı küçük bir azınlık sınıfında sentetik veri üretmek (SMOTE), çapraz doğrulama döngülerinde veri sızıntısına (Data Leakage) ve yapay bir overfitting'e yol açar. Bu yüzden veri çoğaltmak yerine, kayıp fonksiyonunda azınlık sınıfının hatalarını daha ağır cezalandıran `auto_class_weights='Balanced'` parametresi kullanılarak model kararlılığı korunmuştur.
* **Soru: Sabit 0.50 eşik değeri (Threshold) neden esnetildi?**
    * *Cevap:* Sınıf dağılımının %81 patojenik olduğu asimetrik veri kümelerinde sabit 0.50 eşiği, yanlış pozitif (False Positive) oranını artırır. Olasılık uzayında yapılan dinamik arama ile F1-Score'u tepe noktasına çıkaran en optimum eşik değeri dinamik olarak belirlenmiştir.