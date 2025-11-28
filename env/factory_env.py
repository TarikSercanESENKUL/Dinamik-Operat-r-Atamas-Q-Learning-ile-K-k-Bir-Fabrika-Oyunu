"""
Fabrika ortamı – dinamik operatör ataması denemesi.

Bu dosyada, 3 vardiyalı basit bir fabrika günü simüle ediyorum. Makinelerin parçayı
işleyebilmesi için operatör atanması gerekiyor ve operatörlerin de vardiya bazında
çalışma sınırları var. Q-learning ajanı, bu ortamın üzerinde deneme–yanılma ile
politika öğreniyor.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List


class FactoryEnv:
    """
    Dinamik operatör ataması için kullanılan fabrika ortamı.

    Kabaca: 4 makine, 6 operatör ve 3 vardiyadan oluşan bir gün düşünülüyor.
    Ajan, boş bir makine olduğunda hangi operatörü oraya vereceğine karar veriyor.
    Operatörlerin vardiya başına kapasite sınırı ve yorgunluk etkisi de işin içinde.
    """
    
    def __init__(self, config: dict, seed: Optional[int] = None):
        """
        Ortamı başlatıyorum.

        Burada config sözlüğünden tüm fabrika ayarlarını (makine sayısı,
        operatör sayısı, vardiya süresi, ödül parametreleri vs.) okuyorum.
        İstenirse seed vererek aynı senaryoyu tekrar üretmek mümkün.
        """
        self.config = config
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        
        # Yapılandırma parametrelerini çıkar
        self.num_machines = config["num_machines"]
        self.num_operators = config["num_operators"]
        self.num_shifts = config["num_shifts"]
        self.shift_length_minutes = config["shift_length_minutes"]
        self.day_duration_minutes = config["day_duration_minutes"]
        self.target_production_per_day = config["target_production_per_day"]
        self.target_production = config.get("target_production", self.target_production_per_day)  # Uyumluluk için
        self.machine_types = config["machine_types"]
        self.machine_priorities = config["machine_priorities"]
        self.skill_matrix = np.array(config["skill_matrix"])
        self.base_process_times = np.array(config["base_process_times"], dtype=np.float32)
        # Her makine tipi için asgari parça işleme süresi (dakika)
        # Config'de tanımlı değilse, varsayılan olarak base_process_times kullanılır.
        self.min_process_times = np.array(
            config.get("min_process_times", config["base_process_times"]),
            dtype=np.float32,
        )
        self.reward_params = config["reward_params"]
        
        # Operatör kapasitesi vardiya başına: şekil (num_operators, num_shifts)
        self.operator_shift_capacity_minutes = np.array(config["operator_shift_capacity_minutes"])
        
        # Makine arıza/bakım parametreleri
        self.breakdown_prob = config.get("machine_breakdown_probability", 0.02)
        self.maintenance_prob = config.get("machine_maintenance_probability", 0.01)
        self.max_breakdown_shifts = config.get("max_breakdown_duration_shifts", 2)
        self.max_maintenance_shifts = config.get("max_maintenance_duration_shifts", 2)
        self.min_breakdown_minutes = config.get("min_breakdown_duration_minutes", 60)
        self.min_maintenance_minutes = config.get("min_maintenance_duration_minutes", 30)
        
        # Operatör yorgunluk parametreleri
        self.fatigue_threshold_ratio = config.get("fatigue_threshold_ratio", 0.8)
        self.fatigue_penalty_scale = config.get("fatigue_penalty_scale", 0.5)
        
        # Durum takibi
        self.current_time_minutes = 0.0
        self.machines = []
        self.operators = []
        self.produced_good_parts = 0
        self.current_machine_id = None
        
        # Tüm meşgul makinelerin işlem sürelerini takip et (paralel işlem için)
        self.machine_processing_remaining = {}  # machine_id -> kalan_süre
        
        # Operatör değiştirme maliyeti için her makinedeki son operatörü takip et
        self.last_operator_per_machine = [None] * self.num_machines
        
        # Operatör çalışma süresi takibi vardiya başına: şekil (num_operators, num_shifts)
        self.operator_work_minutes_per_shift = np.zeros((self.num_operators, self.num_shifts), dtype=np.float32)
        
        # Görselleştirme için history kaydı
        self.record_history: bool = False
        self.history_frames: List[Dict[str, Any]] = []
        self.last_history_time: float = 0.0  # Son history kaydı zamanı
        self.history_interval_minutes: float = 0.5  # History kaydı aralığı (dakika) - paralel işlemeyi görmek için çok sık
        
        # Ödül şekillendirme için kilometre taşı takibi
        self.reached_50_percent = False
        self.reached_80_percent = False
        
        # Makine arıza/bakım takibi
        # Her makine için: {"status": "idle"/"busy"/"broken"/"maintenance", "down_until": dakika}
        self.machine_down_until = [None] * self.num_machines  # None veya dakika cinsinden bitiş zamanı
        
        # Operatör yorgunluk takibi
        self.operator_fatigue_level = np.zeros(self.num_operators, dtype=np.float32)  # 0.0 (yorgun değil) - 1.0 (çok yorgun)
    
    def _record_snapshot(self) -> None:
        """
        Eğer history kaydı açıksa, o anki durumu GIF için küçük bir snapshot olarak saklıyorum.

        Özellikle operatörleri, makineden ayrılmadan hemen önce yakalamaya çalışıyorum ki
        animasyonda kim hangi makinede çalışıyor daha net görülsün.
        """
        if not self.record_history:
            return
        
        # Mevcut self.machines'e göre machine_assignments, operator_skills ve makine durumları oluştur
        machine_assignments = []
        operator_skills = []
        machine_statuses = []
        
        for m in self.machines:
            # Makine durumu (idle / busy / broken / maintenance)
            status = m.get("status", "idle")
            machine_statuses.append(status)
            
            if m.get("current_operator_id") is not None:
                op_id = m["current_operator_id"]
                machine_assignments.append(op_id)
                machine_type_idx = m["machine_type_index"]
                skill = float(self.skill_matrix[op_id, machine_type_idx])
                operator_skills.append(skill)
            else:
                machine_assignments.append(-1)
                operator_skills.append(-1.0)
        
        snapshot = {
            "time": float(self.current_time_minutes),
            "shift_index": int(self.current_shift_index),
            "machine_assignments": machine_assignments,
            "operator_skills": operator_skills,
            "machine_statuses": machine_statuses,  # Her makinenin gerçek durumu
            "produced_good_parts": int(self.produced_good_parts),
        }
        
        self.history_frames.append(snapshot)
    
    @property
    def current_shift_index(self) -> int:
        """
        Mevcut simülasyon zamanına göre hangi vardiyada olduğumuzu döndürür (0, 1 veya 2).
        """
        shift_idx = int(self.current_time_minutes // self.shift_length_minutes)
        return min(shift_idx, self.num_shifts - 1)
        
    def reset(self, record_history: bool = False) -> Tuple:
        """
        Ortamı ilk güne geri sarmak gibi düşünebiliriz; tüm sayaçları ve makineleri sıfırlar.

        record_history=True verilirse, bu bölümün adımlarını daha sonra GIF çizmek için
        hafızada tutuyorum.
        """
        self.current_time_minutes = 0.0
        self.produced_good_parts = 0
        
        # Makineleri başlat
        self.machines = []
        for i in range(self.num_machines):
            machine = {
                "status": "idle",
                "machine_type_index": i % len(self.machine_types),
                "priority": self.machine_priorities[i] if i < len(self.machine_priorities) else 0,
                "current_operator_id": None,
                "time_remaining": 0.0
            }
            self.machines.append(machine)
        
        # Operatörleri başlat
        self.operators = []
        for i in range(self.num_operators):
            operator = {
                "status": "idle",
                "current_machine_id": None
            }
            self.operators.append(operator)
        
        # Son operatör takibini sıfırla
        self.last_operator_per_machine = [None] * self.num_machines
        
        # Operatör çalışma süresi takibini vardiya başına sıfırla
        self.operator_work_minutes_per_shift = np.zeros((self.num_operators, self.num_shifts), dtype=np.float32)
        
        # History kayıt kurulumu
        self.record_history = record_history
        self.history_frames = []
        self.last_history_time = 0.0
        
        # Kilometre taşı takip bayraklarını sıfırla
        self.reached_50_percent = False
        self.reached_80_percent = False
        
        # İşlem takibini sıfırla
        self.machine_processing_remaining = {}
        
        # Arıza/bakım takibini sıfırla
        self.machine_down_until = [None] * self.num_machines
        
        # Yorgunluk takibini sıfırla
        self.operator_fatigue_level = np.zeros(self.num_operators, dtype=np.float32)
        
        # İlk boş makineyi seç
        self.current_machine_id = self._select_next_idle_machine()
        
        return self._get_state()
    
    def start_recording(self) -> None:
        """Görselleştirme için olay geçmişini kaydetmeye başlar."""
        self.record_history = True
        self.history_frames = []
    
    def stop_recording(self) -> None:
        """Olay geçmişini kaydetmeyi durdurur."""
        self.record_history = False
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Kaydedilen olay geçmişini alır.
        
        Returns:
            History snapshot'larının listesi
        """
        return list(self.history_frames)
    
    def _get_state(self) -> Tuple:
        """
        Mevcut ayrık durum temsilini alır.
        
        3 vardiya simülasyonunu desteklemek için artık mevcut vardiya indeksini içerir.
        
        Returns:
            Mevcut durumu temsil eden hash edilebilir bir tuple
        """
        if self.current_machine_id is None:
            # Boş makine yok, varsayılan bir durum döndür
            current_machine_id = 0
            current_machine_priority = 0
        else:
            current_machine_id = self.current_machine_id
            current_machine_priority = self.machines[current_machine_id]["priority"]
        
        # Mevcut vardiya indeksi (0'dan num_shifts-1'e kadar)
        current_shift_index = self.current_shift_index
        
        # Kalan süre kovası (0-3) - artık tam gün süresine göre
        time_remaining = self.day_duration_minutes - self.current_time_minutes
        if time_remaining <= 0:
            time_bucket = 0
        elif time_remaining <= self.day_duration_minutes / 4:
            time_bucket = 1
        elif time_remaining <= self.day_duration_minutes / 2:
            time_bucket = 2
        else:
            time_bucket = 3
        
        # Üretim açığı kovası (0-3) - artık günlük hedefe göre
        production_gap = self.target_production_per_day - self.produced_good_parts
        if production_gap <= 0:
            gap_bucket = 0
        elif production_gap <= 30:
            gap_bucket = 1
        elif production_gap <= 60:
            gap_bucket = 2
        else:
            gap_bucket = 3
        
        # Operatör müsaitliği (operatör başına 0/1 tuple)
        operator_availability = tuple(
            1 if op["status"] == "idle" else 0
            for op in self.operators
        )
        
        # Mevcut makine için operatör beceri seviyesi kovaları
        if self.current_machine_id is not None:
            machine_type_idx = self.machines[self.current_machine_id]["machine_type_index"]
            operator_skill_buckets = []
            for op_idx in range(self.num_operators):
                skill = self.skill_matrix[op_idx, machine_type_idx]
                if skill < 0.3:
                    skill_bucket = 0  # düşük
                elif skill < 0.7:
                    skill_bucket = 1  # orta
                else:
                    skill_bucket = 2  # yüksek
                operator_skill_buckets.append(skill_bucket)
        else:
            operator_skill_buckets = [0] * self.num_operators
        
        # Makine durumları: 0=idle, 1=busy, 2=broken, 3=maintenance
        machine_status_buckets = []
        for m_id in range(self.num_machines):
            m = self.machines[m_id]
            if m["status"] == "broken":
                status_bucket = 2
            elif m["status"] == "maintenance":
                status_bucket = 3
            elif m["status"] == "busy":
                status_bucket = 1
            else:
                status_bucket = 0  # idle
            machine_status_buckets.append(status_bucket)
        
        # Durum tuple'ına birleştir (artık vardiya indeksini ve makine durumlarını içerir)
        state = (
            current_machine_id,
            current_machine_priority,
            current_shift_index,
            time_bucket,
            gap_bucket,
            operator_availability,
            tuple(operator_skill_buckets),
            tuple(machine_status_buckets),  # Makine durumları eklendi
        )
        
        return state
    
    def _select_next_idle_machine(self) -> Optional[int]:
        """
        Karar verme için bir sonraki boş makineyi seçer.
        Arızalı veya bakımda olan makineleri hariç tutar.
        
        Returns:
            Seçilen boş makinenin makine ID'si veya boş makine yoksa None
        """
        idle_machines = []
        for i, m in enumerate(self.machines):
            # Sadece boş VE arızasız/bakımsız makineleri seç
            if m["status"] == "idle":
                # Arıza/bakım kontrolü
                is_down = self.machine_down_until[i] is not None and self.current_time_minutes < self.machine_down_until[i]
                if not is_down:
                    idle_machines.append(i)
        
        if not idle_machines:
            return None
        
        # Önceliğe göre sırala (azalan), sonra indekse göre
        idle_machines.sort(key=lambda i: (-self.machines[i]["priority"], i))
        
        return idle_machines[0]
    
    def _check_machine_breakdown(self, machine_id: int) -> bool:
        """
        Bir makinenin arıza yapıp yapmadığını kontrol eder.
        
        Args:
            machine_id: Kontrol edilecek makine ID'si
        
        Returns:
            True ise arıza yaptı, False ise normal
        """
        if self.rng.random() < self.breakdown_prob:
            # Arıza süresini hesapla (1-2 vardiya arası, minimum 60 dakika)
            max_duration = self.max_breakdown_shifts * self.shift_length_minutes
            duration = max(
                self.min_breakdown_minutes,
                self.rng.uniform(0.5 * max_duration, max_duration)
            )
            self.machine_down_until[machine_id] = self.current_time_minutes + duration
            self.machines[machine_id]["status"] = "broken"
            return True
        return False
    
    def _check_machine_maintenance(self, machine_id: int) -> bool:
        """
        Bir makinenin bakıma girip girmediğini kontrol eder.
        
        Args:
            machine_id: Kontrol edilecek makine ID'si
        
        Returns:
            True ise bakıma girdi, False ise normal
        """
        if self.rng.random() < self.maintenance_prob:
            # Bakım süresini hesapla (1-2 vardiya arası, minimum 30 dakika)
            max_duration = self.max_maintenance_shifts * self.shift_length_minutes
            duration = max(
                self.min_maintenance_minutes,
                self.rng.uniform(0.5 * max_duration, max_duration)
            )
            self.machine_down_until[machine_id] = self.current_time_minutes + duration
            self.machines[machine_id]["status"] = "maintenance"
            return True
        return False
    
    def _update_operator_fatigue(self, operator_id: int, current_shift_idx: int) -> float:
        """
        Operatör yorgunluğunu günceller ve yorgunluk seviyesini döndürür.
        
        Args:
            operator_id: Operatör ID'si
            current_shift_idx: Mevcut vardiya indeksi
        
        Returns:
            Yorgunluk seviyesi (0.0 - 1.0)
        """
        capacity = self.operator_shift_capacity_minutes[operator_id, current_shift_idx]
        worked = self.operator_work_minutes_per_shift[operator_id, current_shift_idx]
        
        if capacity > 0:
            usage_ratio = worked / capacity
            # Kapasitenin %80'ini aştığında yorgunluk başlar
            if usage_ratio >= self.fatigue_threshold_ratio:
                # Yorgunluk seviyesi: 0.0 (eşik) - 1.0 (kapasite aşımı)
                fatigue = min(1.0, (usage_ratio - self.fatigue_threshold_ratio) / (1.0 - self.fatigue_threshold_ratio))
                self.operator_fatigue_level[operator_id] = fatigue
                return fatigue
        
        return 0.0
    
    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict[str, Any]]:
        """
        Ortamda bir adım çalıştırır.
        
        Bu versiyon paralel işlemeyi destekler: tüm meşgul makineler çalışmaya devam eder
        ve bir sonraki makine bitene kadar (veya yeni bir atama yapılana kadar) işleriz.
        
        Args:
            action: [0, num_operators] aralığında bir tamsayı:
                - 0..num_operators-1: mevcut boş makineye operatör i atama
                - num_operators: atama yapma (makineyi boş bırak)
        
        Returns:
            (next_state, reward, done, info) tuple'ı
        """
        reward = 0.0
        
        # Önce, eylemi işle (varsa boş makineye operatör atama)
        if self.current_machine_id is not None:
            machine = self.machines[self.current_machine_id]
            
            if action < self.num_operators:
                # Makineye operatör atama
                operator_id = action
                
                # Operatörün müsait olup olmadığını kontrol et
                if self.operators[operator_id]["status"] == "busy":
                    # Operatör meşgul, atama yapılamaz - küçük ceza
                    reward -= 0.1
                else:
                    # Varsa operatörü önceki makineden serbest bırak
                    prev_machine_id = self.operators[operator_id]["current_machine_id"]
                    if prev_machine_id is not None:
                        prev_machine = self.machines[prev_machine_id]
                        prev_machine["current_operator_id"] = None
                        prev_machine["status"] = "idle"
                        if prev_machine_id in self.machine_processing_remaining:
                            del self.machine_processing_remaining[prev_machine_id]
                    
                    # Mevcut makineye operatör atama
                    machine["current_operator_id"] = operator_id
                    machine["status"] = "busy"
                    self.operators[operator_id]["current_machine_id"] = self.current_machine_id
                    self.operators[operator_id]["status"] = "busy"
                    
                    # İşlem süresini hesapla ve takibe başla (asgari süreyi zorla)
                    machine_type_idx = machine["machine_type_index"]
                    skill = self.skill_matrix[operator_id, machine_type_idx]
                    raw_time = self.base_process_times[machine_type_idx] / max(skill, 0.1)
                    min_time = self.min_process_times[machine_type_idx]
                    process_time = max(raw_time, min_time)
                    self.machine_processing_remaining[self.current_machine_id] = process_time
                    
                    # Operatör atama anında snapshot kaydet (tüm makineleri görmek için)
                    # NOT: Bu snapshot, yeni atama yapıldıktan SONRA kaydediliyor
                    # Böylece hem yeni atama hem de diğer meşgul makineler görünüyor
                    if self.record_history:
                        self._record_snapshot()
                        self.last_history_time = self.current_time_minutes
                    
                    # Operatör değiştirme maliyetini kontrol et
                    if self.last_operator_per_machine[self.current_machine_id] is not None:
                        if self.last_operator_per_machine[self.current_machine_id] != operator_id:
                            reward -= self.reward_params["penalty_switch_operator"]
                    
                    self.last_operator_per_machine[self.current_machine_id] = operator_id
        
        # Ek: Diğer boş makineleri de otomatik doldur (heuristic), böylece 4 makineye kadar paralel çalışma olur.
        # Bu otomatik atama, ajanın seçmediği makineler için sadece ortam mantığıdır;
        # Q-learning ajanı yalnızca current_machine_id için karar verir. Bu nedenle
        # burada ekstra büyük ödüller vermemeye dikkat ediyoruz; aksi halde return
        # grafiği erken bölümlerde gereksiz sıçramalar yapabiliyor.
        if self.record_history:
            # Otomatik atama yapmadan önce mevcut durumu kaydetmek gereksiz; zaten yukarıda kaydettik.
            pass
        
        # Müsait operatörleri ve boş makineleri topla
        available_operators = [
            op_id for op_id, op in enumerate(self.operators)
            if op["status"] == "idle"
        ]
        
        idle_machines = []
        for m_id, m in enumerate(self.machines):
            if m["status"] == "idle":
                # Arıza/bakım durumunda olan makineleri hariç tut
                is_down = self.machine_down_until[m_id] is not None and self.current_time_minutes < self.machine_down_until[m_id]
                if not is_down:
                    idle_machines.append(m_id)
        
        # current_machine_id zaten ajan tarafından atandıysa, onu otomatik atamadan çıkar
        if self.current_machine_id is not None and self.machines[self.current_machine_id]["status"] == "busy":
            idle_machines = [m_id for m_id in idle_machines if m_id != self.current_machine_id]
        
        # Boş makineleri önceliğe göre sırala (yüksek öncelik önce)
        idle_machines.sort(key=lambda i: (-self.machines[i]["priority"], i))
        
        # Her boş makine için en uygun (müsait) operatörü seç ve ata
        for m_id in idle_machines:
            if not available_operators:
                break
            
            machine = self.machines[m_id]
            machine_type_idx = machine["machine_type_index"]
            
            # En yüksek beceriye sahip operatörü seç (müsaitler arasından)
            best_op = None
            best_skill = -1.0
            for op_id in available_operators:
                skill = float(self.skill_matrix[op_id, machine_type_idx])
                if skill > best_skill:
                    best_skill = skill
                    best_op = op_id
            
            if best_op is None:
                break
            
            # Operatörü bu makineye ata (görselleştirme ve daha dolu fabrika için)
            machine["current_operator_id"] = best_op
            machine["status"] = "busy"
            self.operators[best_op]["current_machine_id"] = m_id
            self.operators[best_op]["status"] = "busy"
            
            # İşlem süresini hesapla ve takibe başla (asgari süreyi zorla)
            raw_time = self.base_process_times[machine_type_idx] / max(best_skill, 0.1)
            min_time = self.min_process_times[machine_type_idx]
            process_time = max(raw_time, min_time)
            self.machine_processing_remaining[m_id] = process_time
            
            # Kullanılan operatörü listeden çıkar
            available_operators.remove(best_op)
        
        # Arıza/bakım süresi biten makineleri tekrar kullanılabilir hale getir
        for m_id in range(self.num_machines):
            if self.machine_down_until[m_id] is not None:
                if self.current_time_minutes >= self.machine_down_until[m_id]:
                    # Arıza/bakım bitti, makineyi tekrar kullanılabilir yap
                    self.machine_down_until[m_id] = None
                    if self.machines[m_id]["status"] in ["broken", "maintenance"]:
                        self.machines[m_id]["status"] = "idle"
        
        # Şimdi, bir sonraki makine bitene kadar zamanı ilerlet (paralel işleme)
        # Tüm meşgul makineler arasında minimum kalan işlem süresini bul
        if not self.machine_processing_remaining:
            # İşleyen makine yok, zamanı biraz ilerlet (ama episode süresini aşma)
            time_advance = min(1.0, self.day_duration_minutes - self.current_time_minutes)
            if time_advance > 0:
                self.current_time_minutes += time_advance
        else:
            # İlk bitecek makineyi bul
            min_time = min(self.machine_processing_remaining.values())
            time_advance = min_time
            
            # Episode süresini aşmayacak şekilde zaman ilerlemesini sınırla
            max_advance = self.day_duration_minutes - self.current_time_minutes
            if max_advance <= 0:
                # Episode zaten bitmiş, daha fazla işleme yapma
                time_advance = 0.0
            else:
                time_advance = min(time_advance, max_advance)
            
            # Zamanı ilerlet
            if time_advance > 0:
                # Zaman ilerlemeden ÖNCE mevcut durumu kaydet (paralel işleme sırasında tüm meşgul makineleri görmek için)
                # Bu snapshot, tüm meşgul makinelerin durumunu içerir
                if self.record_history:
                    # Tüm meşgul makinelerin durumunu kaydet
                    self._record_snapshot()
                    self.last_history_time = self.current_time_minutes
                
                self.current_time_minutes += time_advance
                
                # Tüm meşgul makinelerin kalan sürelerini güncelle
                finished_machines = []
                for m_id, remaining in list(self.machine_processing_remaining.items()):
                    new_remaining = remaining - time_advance
                    if new_remaining <= 0.0001:  # Makine bitti
                        finished_machines.append(m_id)
                        self.machine_processing_remaining[m_id] = 0.0
                    else:
                        self.machine_processing_remaining[m_id] = new_remaining
            else:
                finished_machines = []
            
            # Tüm biten makineleri işle (sadece episode bitmemişse)
            # Episode bitmişse, biten makineleri işleme (zaman aşımı nedeniyle)
            episode_ended = self.current_time_minutes >= self.day_duration_minutes
            
            for m_id in finished_machines:
                # Episode bitmişse, bu makineyi işleme (zaman aşımı)
                if episode_ended:
                    # Makineyi ve operatörü serbest bırak ama parça üretme
                    m = self.machines[m_id]
                    if m["current_operator_id"] is not None:
                        operator_id = m["current_operator_id"]
                        m["status"] = "idle"
                        m["current_operator_id"] = None
                        self.operators[operator_id]["status"] = "idle"
                        self.operators[operator_id]["current_machine_id"] = None
                        if m_id in self.machine_processing_remaining:
                            del self.machine_processing_remaining[m_id]
                    continue
                
                m = self.machines[m_id]
                if m["current_operator_id"] is not None:
                    operator_id = m["current_operator_id"]
                    machine_type_idx = m["machine_type_index"]
                    skill = self.skill_matrix[operator_id, machine_type_idx]
                    
                    # Mevcut vardiya indeksini al
                    current_shift_idx = self.current_shift_index
                    
                    # Operatör çalışma süresini takip et (asgari işlem süresini dikkate al)
                    raw_time = self.base_process_times[machine_type_idx] / max(skill, 0.1)
                    min_time = self.min_process_times[machine_type_idx]
                    process_time = max(raw_time, min_time)
                    self.operator_work_minutes_per_shift[operator_id, current_shift_idx] += process_time
                    
                    # Yorgunluk kontrolü ve güncelleme
                    fatigue = self._update_operator_fatigue(operator_id, current_shift_idx)
                    if fatigue > 0:
                        # Yorgunluk cezası: yorgunluk seviyesine göre
                        fatigue_penalty = self.fatigue_penalty_scale * fatigue
                        reward -= fatigue_penalty
                    
                    # Kapasite aşımını kontrol et
                    capacity = self.operator_shift_capacity_minutes[operator_id, current_shift_idx]
                    worked_minutes = self.operator_work_minutes_per_shift[operator_id, current_shift_idx]
                    
                    if worked_minutes > capacity:
                        overuse_ratio = (worked_minutes - capacity) / capacity
                        penalty = self.reward_params.get("penalty_over_capacity", 1.0) * overuse_ratio
                        reward -= penalty
                    
                    # Beceri bazlı ödüller
                    # Uygun eşleşme kontrolü (yüksek beceri = uygun eşleşme)
                    if skill >= 0.7:
                        reward += self.reward_params.get("reward_appropriate_assignment", 10.0)
                    elif skill < 0.3:
                        # Düşük beceri = uygunsuz eşleşme, yavaş üretim
                        reward -= self.reward_params.get("penalty_slow_production", 5.0)
                        reward -= self.reward_params["penalty_mismatch_low_skill"]
                    else:
                        # Orta beceri için küçük ödül
                        reward += self.reward_params["reward_skill_scale"] * skill
                    
                    # Parçanın hatalı olup olmadığını belirle
                    p_defect = max(0.0, 0.5 - skill)
                    is_defective = self.rng.random() < p_defect
                    
                    if is_defective:
                        # Hatalı ürün üretme cezası (artırıldı)
                        reward -= self.reward_params.get("penalty_defective_product", 15.0)
                    else:
                        # Başarılı ürün üretimi
                        reward += self.reward_params["reward_per_good_part"]
                        reward += self.reward_params.get("reward_successful_part", 1.0)
                        self.produced_good_parts += 1
                        
                        # Kilometre taşı bonusları
                        target = self.target_production_per_day
                        bonus_50 = self.reward_params.get("bonus_reach_50_percent", 0.0)
                        bonus_80 = self.reward_params.get("bonus_reach_80_percent", 0.0)
                        
                        if not self.reached_50_percent and self.produced_good_parts >= 0.5 * target:
                            reward += bonus_50
                            self.reached_50_percent = True
                        
                        if not self.reached_80_percent and self.produced_good_parts >= 0.8 * target:
                            reward += bonus_80
                            self.reached_80_percent = True
                    
                    # Serbest bırakmadan ÖNCE history snapshot'ı kaydet
                    self._record_snapshot()
                    
                    # Makine arıza/bakım kontrolü (işlem bitince)
                    # Önce arıza kontrolü (daha ciddi)
                    if self._check_machine_breakdown(m_id):
                        # Arıza yaptı, makineyi serbest bırak ama operatörü de serbest bırak
                        m["current_operator_id"] = None
                        self.operators[operator_id]["status"] = "idle"
                        self.operators[operator_id]["current_machine_id"] = None
                        del self.machine_processing_remaining[m_id]
                        continue  # Arıza durumunda makine kullanılamaz
                    # Sonra bakım kontrolü
                    elif self._check_machine_maintenance(m_id):
                        # Bakıma girdi, makineyi serbest bırak ama operatörü de serbest bırak
                        m["current_operator_id"] = None
                        self.operators[operator_id]["status"] = "idle"
                        self.operators[operator_id]["current_machine_id"] = None
                        del self.machine_processing_remaining[m_id]
                        continue  # Bakım durumunda makine kullanılamaz
                    
                    # Normal durum: Operatörü ve makineyi serbest bırak
                    m["status"] = "idle"
                    m["current_operator_id"] = None
                    self.operators[operator_id]["status"] = "idle"
                    self.operators[operator_id]["current_machine_id"] = None
                    del self.machine_processing_remaining[m_id]
        
        # Boş makine cezası - sadece kullanılabilir makineler için
        idle_count = 0
        for i, m in enumerate(self.machines):
            is_down = self.machine_down_until[i] is not None and self.current_time_minutes < self.machine_down_until[i]
            if m["status"] == "idle" and not is_down:
                idle_count += 1
        reward -= self.reward_params.get("penalty_machine_idle", 10.0) * idle_count
        
        # Makine boş kalmadan önce operatör atama ödülü (eğer boş makine yoksa)
        if idle_count == 0 and len([m for m in self.machines if m["status"] == "busy"]) > 0:
            reward += self.reward_params.get("reward_prevent_idle", 5.0)
        
        # Episode bitip bitmediğini kontrol et
        done = self.current_time_minutes >= self.day_duration_minutes
        
        # Terminal ödül ayarlaması
        if done:
            if self.produced_good_parts >= self.target_production_per_day:
                reward += self.reward_params["goal_bonus"]
            else:
                shortfall = self.target_production_per_day - self.produced_good_parts
                reward -= self.reward_params["shortfall_penalty_scale"] * shortfall
        
        # NOT: Otomatik operatör atamaları kaldırıldı
        # Tüm kararlar ajan tarafından verilmeli (Q-learning prensibi)
        # Ajan her boş makine için sırayla karar verir
        
        # Ajan kararı için bir sonraki boş makineyi seç (varsa, episode bitmemişse)
        if not done:
            self.current_machine_id = self._select_next_idle_machine()
        else:
            self.current_machine_id = None
        
        # Periyodik olarak snapshot kaydet (tüm makinelerin durumunu görmek için)
        if self.record_history:
            # Belirli zaman aralıklarında veya önemli olaylarda kaydet
            time_since_last = self.current_time_minutes - self.last_history_time
            if time_since_last >= self.history_interval_minutes or done:
                self._record_snapshot()
                self.last_history_time = self.current_time_minutes
        
        # Bir sonraki durumu al
        next_state = self._get_state()
        
        info = {
            "produced_good_parts": self.produced_good_parts,
            "current_time": self.current_time_minutes,
            "current_shift": self.current_shift_index
        }
        
        return next_state, reward, done, info
