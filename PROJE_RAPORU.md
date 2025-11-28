## Fabrikada Dinamik Operatör Ataması  
### Tabular Q-Learning ile Kısa Proje Raporu

Bu çalışmada, 4 makine ve 6 operatörlü küçük bir üretim hattında,  
operatör–makine atamasını **tabular Q-learning** ile otomatikleştirmeyi denedim.  
Amaç; operatör becerileri, vardiya süreleri, arıza/bakım durumları ve günlük üretim hedefi  
gibi gerçekçi kısıtlar altında, ajanın zamanla mantıklı bir atama politikası öğrenmesidir.

---

## 1. Problem ve Senaryo

- **Makineler:** Pres, Torna, Kaynak, Paketleme (toplam 4 makine).  
- **Operatörler:** 6 kişi, her makinede farklı beceri seviyeleri (0.1–1.0 arası).  
- **Zaman:** 3 vardiya × 8 saat = 1440 dakika (bir gün).  
- **Hedef:** Günde ortalama **90 sağlam parça** üretmek.  
- **Kısıtlar:**
  - Her operatörün vardiya başına dakika sınırı (yorgunluk/kapasite).  
  - Makinelerde rastgele **arıza** ve **bakım** süreleri.  
  - Yüksek becerili operatör + uygun makine = daha hızlı ve az hatalı üretim.

Pratikte cevap aranan soru şudur:  
**“Hangi anda, hangi boş makineye, hangi operatörü verirsem uzun vadede daha iyi üretim alırım?”**

---

## 2. RL Formülasyonu (State – Action – Reward)

### 2.1 State (Durum)

Her karar anında ajan, fabrikanın “özet bir fotoğrafını” görüyor:

- Atama bekleyen makinenin ID’si ve önceliği (0–3, öncelik 0–2).  
- Hangi vardiyada olduğumuz (0, 1, 2) ve günün bitimine kalan süre kovası (0–3).  
- Hedefe kalan parça miktarının kovası (0–3).  
- Her operatör için **boş / meşgul** bilgisi (0/1).  
- Mevcut makineye göre her operatörün beceri kovası (düşük / orta / yüksek).  
- Her makinenin durumu: boşta, çalışıyor, arızalı veya bakımda.

Bu bilgiler bir `tuple` içinde birleştirilip Q-tablosunda state anahtarı olarak kullanılıyor.

### 2.2 Action (Eylem)

Eylem uzayı basit tutuldu:

- `0 .. 5` → ilgili operatörü (0–5) şu anki boş makineye ata.  
- `6`      → bu adımda kimseyi atama, makineyi boş bırak.  

Toplam **7 eylem** vardır. Ajan her karar anında bu 7 seçenekten birini seçer.

### 2.3 Reward (Ödül)

Ödül fonksiyonu; **sağlam parça üretimini artıran, makine boşluğunu azaltan  
ve hatalı üretimi engelleyen** kararları ödüllendirecek şekilde tasarlandı:

- Sağlam parça üretmek → pozitif ödül (parça başına).  
- Yüksek becerili operatörü doğru makineye atamak → ek bonus.  
- Makineyi gereksiz boş bırakmak → ceza.  
- Hatalı ürün ve çok düşük beceriyle yavaş üretim → daha büyük ceza.  
- Gün sonu:
  - Hedefe yaklaştıkça bonus,  
  - Hedefin çok altında kalındığında eksik parça başına ceza.

Bu yapı sayesinde ajan, uzun vadede **“doğru kişiyi doğru makineye ver, makineleri boş bırakma,
çok hata yapma”** tarzı bir politika öğrenmeye başlıyor.

---

## 3. Ortam ve Ajanın Kısa Özeti

### 3.1 Ortam (FactoryEnv)

`FactoryEnv` sınıfı:

- Zamanı dakikalar üzerinden takip eder, gün 1440 dakikada biter.  
- Aynı anda birden fazla makinenin çalışabildiği **paralel** bir üretim süreci simüle eder.  
- Her operatörün vardiya başına çalışma süresini ve yorgunluk seviyesini tutar.  
- Her parça sonunda, arıza/bakım durumlarını kontrol eder ve makineyi geçici olarak devre dışı bırakabilir.  
- Ajanın verdiği atama kararlarına göre ödül/ceza üretir ve bir sonraki state’i hesaplar.

### 3.2 Q-Learning Ajanı

`QLearningAgent` sınıfı:

- Q-tablosunu `dict[state][action]` yapısında tutar.  
- Eylem seçiminde epsilon-greedy stratejisi kullanır:
  - Başta `epsilon ≈ 1.0` (fazla keşif),  
  - Eğitim boyunca yaklaşık **10.000 bölüm** içinde `0.05` seviyesine kadar düşer.  
- Öğrenme oranı (alpha) ilk başta yüksek, sonra yavaş yavaş azalır; böylece
  erken dönemlerde hızlı değişim, geç dönemlerde daha stabil bir öğrenme elde edilir.  
- Q-değerlerini klasik formülle günceller:
  - \( Q(s,a) ← Q(s,a) + α (r + γ \max_{a'} Q(s',a') − Q(s,a)) \).

---

## 4. Eğitim, Değerlendirme ve Görselleştirme

- Eğitim script’i: `scripts/main_train.py`  
  - 10.000 bölüm boyunca ajan ortamda eğitilir.  
  - Q-tablosu hem `.pkl` hem de `.h5` formatında kaydedilir.  
  - En iyi bölümlerden biri kullanılarak **eğitim GIF’i** üretilir:
    - `outputs/training_best_episode.gif`

- Değerlendirme script’i: `scripts/main_eval.py`  
  - Eğitilmiş ajan, 100 bölüm boyunca **greedy** politika ile çalıştırılır.  
  - Ortalama ödül ve ortalama üretim raporlanır.  
  - İlk bölümün akışı için `outputs/evaluation_run.gif` üretilir.

- Test script’i: `scripts/main_test.py`  
  - Yine 100 bölüm greedy test yapılır, daha ayrıntılı istatistikler toplanır.  
  - İlk test bölümünden `outputs/test_run.gif` elde edilir.

Eğitim sürecinin genel eğilimini görmek için:

- `outputs/returns.png` → bölüm başına toplam ödül (hareketli ortalama ile).  
- `outputs/productions.png` → bölüm başına üretilen sağlam parça sayısı.

Bu grafikler, ajan gerçekten öğreniyor mu ve ne zaman “plato” seviyesine oturuyor,
onu anlamayı kolaylaştırıyor.

---

## 5. Sonuç ve Kısa Değerlendirme

Bu proje, görece küçük ama gerçekçi bir fabrika senaryosunda,
**tabular Q-learning** ile dinamik operatör atamasının nasıl yapılabileceğini göstermektedir.

- Ajan; makine önceliği, operatör becerileri, kapasite kısıtları ve arıza/bakım gibi  
  unsurları dolaylı olarak state ve reward üzerinden “öğrenebilmektedir”.  
- Eğitim ilerledikçe, makinelerin daha az boş kaldığı ve sağlam parça üretiminin
  hedefe yaklaştığı gözlemlenmektedir.  
- Kod yapısı özellikle öğrenci seviyesinde sade tutulduğu için,  
  Q-learning adımlarını ve tablo güncellemelerini takip etmek görece kolaydır.

Genel olarak, bu çalışma hem pekiştirmeli öğrenme kavramlarını uygulamalı olarak  
denemek, hem de üretim planlama problemlerine yapay zekâ bakış açısından  
yaklaşmak adına faydalı bir deneyim olmuştur.


