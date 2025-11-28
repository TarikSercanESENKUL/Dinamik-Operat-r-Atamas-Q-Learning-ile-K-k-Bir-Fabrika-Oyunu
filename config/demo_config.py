"""
Fabrika simülasyonu için yapılandırma modülü.

Bu modül, fabrika ortamının tüm parametrelerini içerir. Makine sayıları,
operatör becerileri, ödül parametreleri vs. burada tanımlanıyor.
"""


def get_demo_config():
    """
    Fabrika simülasyonu için demo yapılandırmasını döndürür.
    
    Bu yapılandırma, vardiya başına operatör kapasite kısıtları ile 3 vardiyalı
    bir fabrika gününü destekler. Fabrika tam bir gün (3 vardiya) çalışır,
    sonra sıfırlanır.
    
    Returns:
        Tüm yapılandırma parametrelerini içeren sözlük
    """
    num_machines = 4
    num_operators = 6  # Daha iyi makine kullanımı için 6 operatör
    num_shifts = 3
    shift_length_minutes = 480  # Her vardiya 8 saat
    day_duration_minutes = num_shifts * shift_length_minutes  # Tam gün = 3 vardiya
    
    # Beceri matrisi: her makine tipinde operatör becerisi
    # Şekil: (num_operators, num_machines)
    # Değerler 0.1 (çok kötü) ile 1.0 (mükemmel) arasında
    # Daha çeşitli beceri seviyeleri: uzman, iyi, orta, kötü, çok kötü
    skill_matrix = [
        [0.95, 0.35, 0.15, 0.20],  # Operatör 0: pres'te uzman, diğerlerinde çok kötü
        [0.25, 0.90, 0.65, 0.55],  # Operatör 1: tornada uzman, diğerlerinde orta-kötü
        [0.55, 0.30, 0.95, 0.85],  # Operatör 2: kaynakta uzman, paketlemede iyi
        [0.45, 0.50, 0.48, 0.52],  # Operatör 3: tüm makinelerde ortalama (çok yönlü)
        [0.70, 0.65, 0.55, 0.92],  # Operatör 4: paketlemede uzman, diğerlerinde iyi
        [0.88, 0.58, 0.42, 0.68],  # Operatör 5: pres'te çok iyi, diğerlerinde orta-iyi
    ]
    
    # Operatör vardiya kapasitesi: operatör başına vardiya başına maksimum çalışma dakikası
    # Şekil: (num_operators, num_shifts)
    # Bazı operatörler "daha güçlü" (yüksek kapasite), bazıları yarı zamanlı veya daha zayıf
    # Kapasiteyi yaklaşık %20 artırdık (vardiya başına ~60 dakika ekleyerek) böylece fabrika
    # fiziksel olarak günde daha fazla parça üretebilir, RL ajanının daha yüksek üretim seviyelerine
    # ulaşmasına izin verir
    operator_shift_capacity_minutes = [
        [480, 460, 480],  # Operatör 0: güçlü, vardiyalar arasında tutarlı (420'den artırıldı)
        [460, 480, 460],  # Operatör 1: güçlü, orta vardiyada biraz daha iyi (400'den artırıldı)
        [440, 420, 440],  # Operatör 2: orta kapasite, orta vardiyada daha zayıf (380/360'dan artırıldı)
        [460, 460, 460],  # Operatör 3: tutarlı ortalama kapasite (400'den artırıldı)
        [480, 440, 480],  # Operatör 4: güçlü, ama orta vardiyada daha zayıf (420/380'den artırıldı)
        [470, 450, 470],  # Operatör 5: güçlü, tutarlı kapasite
    ]
    
    config = {
        # Temel fabrika parametreleri
        "num_machines": num_machines,  # Fabrikadaki makine sayısı
        "num_operators": num_operators,  # Müsait operatör sayısı
        
        # Vardiya yapılandırması
        "num_shifts": num_shifts,  # Günlük vardiya sayısı (3 vardiya: sabah, öğleden sonra, gece)
        "shift_length_minutes": shift_length_minutes,  # Her vardiyanın dakika cinsinden süresi (8 saat)
        "day_duration_minutes": day_duration_minutes,  # Toplam gün süresi = num_shifts * shift_length_minutes
        
        # Üretim hedefleri
        "target_production_per_day": 90,  # Tam gün (3 vardiya) başına üretilecek iyi parça sayısı hedefi
        "target_production": 90,  # Geriye dönük uyumluluk için takma ad (artık günlük hedefi temsil ediyor, vardiya başına değil)
        
        # Makine yapılandırması
        "machine_types": ["press", "lathe", "welding", "packing"],  # Makine tipleri
        "machine_priorities": [1, 2, 1, 0],  # Makine başına öncelik seviyeleri (0=düşük, 1=orta, 2=yüksek)
        # Daha yüksek öncelikli makinelere önce operatör atanmalı
        
        # Operatör beceri matrisi
        # skill_matrix[i][j] = operatör i'nin makine tipi j'deki becerisi
        # Yüksek beceri daha hızlı işleme ve daha düşük hata oranı anlamına gelir
        "skill_matrix": skill_matrix,
        
        # Vardiya başına operatör kapasitesi
        # operator_shift_capacity_minutes[i][j] = operatör i için vardiya j'de maksimum çalışma dakikası
        # Kapasiteyi aşmak yorgunluk cezalarına yol açar
        # Bazı operatörler daha güçlüdür (yüksek kapasite), bazıları yarı zamanlıdır (belirli vardiyalarda düşük kapasite)
        "operator_shift_capacity_minutes": operator_shift_capacity_minutes,
        
        # Her makine tipi için temel işlem süreleri (parça başına dakika)
        # Gerçek işlem süresi = temel_süre / operatör_becerisi
        # Not: Aşağıdaki minimum işlem süreleri ile birlikte kullanılır; böylece
        # beceri ne kadar yüksek olursa olsun bir parçanın işlenmesi fiziksel
        # olarak belirli bir süreden daha kısa sürmez.
        "base_process_times": [6.0, 7.0, 9.0, 5.0],  # pres, torna, kaynak, paketleme

        # Her makine tipi için **asgari parça işleme süreleri** (dakika cinsinden)
        # Bu projenin senaryosuna göre minimum süreler:
        # - Pres:       10 dakika
        # - Torna:      45 dakika
        # - Kaynak:     1 saat 15 dakika (75 dakika)
        # - Paketleme:  25 dakika
        #
        # Sıralama makine tiplerine şu şekilde eşlenmiştir:
        #   pres      → 10 dk
        #   torna     → 45 dk
        #   kaynak    → 75 dk
        #   paketleme → 25 dk
        #
        # Yani, operatör becerisi ne kadar yüksek olursa olsun, ilgili makinede
        # bir parçayı bu süreden daha hızlı işleyemez.
        "min_process_times": [10.0, 45.0, 75.0, 25.0],
        
        # Makine arıza/bakım parametreleri
        "machine_breakdown_probability": 0.02,  # Her işlem sonrası arıza olasılığı (2%)
        "machine_maintenance_probability": 0.01,  # Her işlem sonrası bakım ihtiyacı olasılığı (1%)
        "max_breakdown_duration_shifts": 2,  # Maksimum arıza süresi (2 vardiya)
        "max_maintenance_duration_shifts": 2,  # Maksimum bakım süresi (2 vardiya)
        "min_breakdown_duration_minutes": 60,  # Minimum arıza süresi (1 saat)
        "min_maintenance_duration_minutes": 30,  # Minimum bakım süresi (30 dakika)
        
        # Operatör yorgunluk parametreleri
        "fatigue_threshold_ratio": 0.8,  # Kapasitenin %80'ini kullandığında yorgunluk başlar
        "fatigue_penalty_scale": 0.5,  # Yorgunluk cezası ölçeği
        
        # Ödül fonksiyonu parametreleri
        # Not: Buradaki sayıları özellikle şöyle ayarladım:
        # - Erken episode'larda (politika kötü iken) toplam ödül nispeten düşük kalsın,
        # - İlerleyen episode'larda ajan hedefe daha çok yaklaştıkça ekstra bonuslar
        #   devreye girsin ve return grafiği yukarı doğru çıksın.
        "reward_params": {
            # Üretilen her iyi parça için pozitif ödül
            # Orta seviyede tuttum; tek başına çok büyük sıçrama yaratmasın diye.
            "reward_per_good_part": 2.0,
            
            # Karar anında boş makine başına ceza
            # Boş makineler için küçük ama anlamlı maliyet
            "penalty_idle_machine": 0.2,
            
            # Operatör becerisi için ödül ölçekleme faktörü
            # Yüksek becerili atamaları daha güçlü ödüllendirmek için artırıldı
            "reward_skill_scale": 0.5,
            
            # Düşük becerili operatör (< 0.3) atama cezası
            # Çok düşük becerili atamalar için net bir ceza tut
            "penalty_mismatch_low_skill": 1.0,
            
            # Aynı makinede operatör değiştirme cezası
            # Orta seviye değiştirme maliyeti
            "penalty_switch_operator": 0.5,
            
            # Hatalı parça üretme cezası
            # Hatalı parçaları daha net cezalandırmak için artırıldı
            "penalty_defect": 10.0,
            
            # Bir vardiyada operatör kapasitesini aşma cezası
            # Operatör kapasitesinden fazla çalıştığında uygulanır, aşım oranına göre ölçeklenir
            "penalty_over_capacity": 1.0,
            
            # Gün sonunda günlük üretim hedefi karşılanırsa bonus ödül
            # Çok aşırı olmasın diye orta seviyeye çektim; hedefe ulaşmak
            # hâlâ anlamlı ama tek seferde grafiği bozmuyor.
            "goal_bonus": 80.0,
            
            # Üretim eksikliği için ceza ölçeği
            # Sağlam parça kalitesine daha fazla odaklanmak için biraz azaltıldı
            "shortfall_penalty_scale": 0.3,
            
            # Ara üretim hedeflerine ulaşmak için kilometre taşı bonusları
            # Bunlar ajanın üretim hedefine yaklaştığında ek sinyaller sağlar
            # Ara kilometre taşı bonusları, sadece hedefin %50 ve %80'ini geçerken episode başına bir kez verilir
            # Günlük hedefin %50'sine ulaşmak için bonus (episode başına ilk kez)
            "bonus_reach_50_percent": 10.0,
            
            # Günlük hedefin %80'ine ulaşmak için bonus (episode başına ilk kez)
            "bonus_reach_80_percent": 20.0,
            
            # Yeni senaryo ödülleri (senaryo açıklamasına göre)
            # Uygun eşleşme ve sağlam parça üretimini güçlü teşvik et
            "reward_appropriate_assignment": 15.0,  # Uygun operatör-makine eşleşmesi ve başarılı üretim
            "penalty_slow_production": 8.0,  # Uygunsuz eşleşme sonucu yavaş üretim (artırıldı)
            "penalty_defective_product": 25.0,  # Hatalı ürün üretme için ağır ceza
            "reward_prevent_idle": 5.0,  # Makine boş kalmadan önce operatör atama
            "penalty_machine_idle": 10.0,  # Makine boş kalma cezası
            # Her başarılı (sağlam) ürün üretiminde ek ödül (parça başına ödüle ek)
            "reward_successful_part": 1.0,
        }
    }
    
    return config
