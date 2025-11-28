"""
Fabrika ortamı animasyonları için görselleştirme yardımcıları.

Bu modül, operatör atamalarını zaman içinde gösteren GIF animasyonları
oluşturmak için yardımcı fonksiyonlar içerir.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Dict, Any


def render_timeline_gif(
    history: List[Dict[str, Any]],
    config: dict,
    output_path: str,
    title: str = "Fabrika Operatör Ataması",
    fps: int = 10,  # Daha hızlı animasyon için artırıldı
) -> None:
    """
    Zaman içinde hangi operatörün hangi makinede çalıştığını gösteren bir GIF animasyonu oluşturur.
    
    Bu görselleştirme şunları gösterir:
    - Alt satırda (y=0) makineler gri kareler olarak M0, M1, M2, ... etiketli
    - Her makine karesi içinde mevcut operatörü (O0, O1, ...) veya "Idle" gösterir
    - Operatörler, makine pozisyonları ve boş/park alanı arasında hareket eden renkli daireler
    - Operatörler makinelerden açıkça ayrılmıştır (y=0.6'da) böylece işçi hareketi görsel olarak net
    - Operatörlerin dairelerinin üstünde bir beceri skoru yazdırılır
    - Daha yüksek beceri → biraz daha büyük daire
    
    Bu, öğrenci dostu bir görselleştirme yardımcısıdır ve ajanın politikasının
    bir gün boyunca operatörleri makinelere nasıl atadığını görmeyi kolaylaştırır.
    
    Args:
        history: FactoryEnv tarafından üretilen snapshot listesi, her biri en az şunları içerir:
            * "time": float - mevcut simülasyon zamanı dakika cinsinden
            * "shift_index": int - mevcut vardiya (0, 1 veya 2)
            * "machine_assignments": list[int] - her makine için operator_id veya boş için -1
            * "operator_skills": list[float] - her makine için beceri skoru veya boş için -1.0
            * "produced_good_parts": int - şu ana kadar üretilen toplam iyi parça
        config: Yapılandırma sözlüğü (num_machines, num_operators vb. bilmek için kullanılır)
        output_path: GIF'i kaydetmek için yer (örn., "outputs/training_run.gif")
        title: Animasyon için başlık string'i
        fps: GIF için saniye başına frame (varsayılan: 8, daha hızlı animasyon için)
    """
    num_machines = config["num_machines"]
    num_operators = config["num_operators"]
    skill_matrix = config.get("skill_matrix", [])
    
    if not history:
        print("Uyarı: Boş history, GIF oluşturulamaz")
        return
    
    # Her operatör için farklı renkler tanımla
    # Görsel olarak farklı basit bir renk paleti kullanıyoruz
    operator_colors = [
        '#FF6B6B',  # Kırmızı
        '#4ECDC4',  # Teal
        '#45B7D1',  # Mavi
        '#FFA07A',  # Açık Somon
        '#98D8C8',  # Nane
        '#F7DC6F',  # Sarı
        '#BB8FCE',  # Mor
        '#85C1E2',  # Gökyüzü Mavisi
    ]
    # Operatör sayısı renk sayısından fazlaysa genişlet
    while len(operator_colors) < num_operators:
        operator_colors.extend(operator_colors)
    
    # Figürü başlat
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.close(fig)  # Etkileşimli olmayan ortamlarda göstermemek için kapat
    
    def update(frame_idx: int):
        """Animasyon frame'leri için güncelleme fonksiyonu."""
        ax.clear()
        
        # Bu frame için snapshot al
        snapshot = history[frame_idx]
        machine_assignments = snapshot["machine_assignments"]
        operator_skills = snapshot.get("operator_skills", [-1.0] * num_machines)
        machine_statuses = snapshot.get("machine_statuses", None)
        current_time = snapshot.get("time", 0.0)
        shift_index = snapshot.get("shift_index", 0)
        produced_parts = snapshot.get("produced_good_parts", 0)
        episode_number = snapshot.get("episode_number", 0)
        
        # Makineleri alt satırda (y=0) gri kareler olarak çiz
        for m_id in range(num_machines):
            # Makine karesini çiz
            rect = plt.Rectangle((m_id - 0.3, -0.3), 0.6, 0.6,
                               facecolor='#cccccc', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Makine adı (M0, M1, ...) - biraz yukarı alınmış
            ax.text(m_id, -0.45, f"M{m_id}", ha='center', va='top',
                   fontsize=10, fontweight='bold')
            
            # Makinenin durumunu belirle (idle / busy / broken / maintenance)
            status_text = ""
            status_color = "gray"
            if machine_statuses is not None and m_id < len(machine_statuses):
                st = machine_statuses[m_id]
                if st == "busy":
                    status_text = "Çalışıyor"
                    status_color = "green"
                elif st == "broken":
                    status_text = "Arızalı"
                    status_color = "red"
                elif st == "maintenance":
                    status_text = "Bakım"
                    status_color = "orange"
                else:
                    status_text = "Boşta"
                    status_color = "gray"
            else:
                # History eski formatta ise, sadece operatör varlığına göre tahmin et
                op_id_tmp = machine_assignments[m_id] if m_id < len(machine_assignments) else -1
                if op_id_tmp == -1:
                    status_text = "Boşta"
                    status_color = "gray"
                else:
                    status_text = "Çalışıyor"
                    status_color = "green"
            
            # Makinenin altına durum etiketini yaz (Boşta / Çalışıyor / Arızalı / Bakım)
            ax.text(m_id, -0.8, status_text, ha='center', va='top',
                   fontsize=8, color=status_color, style='italic')
            
            # Makine karesi içinde mevcut operatörü veya "Idle" göster
            op_id = machine_assignments[m_id] if m_id < len(machine_assignments) else -1
            if op_id == -1:
                # Makine boş (duruma göre Boşta/Arızalı/Bakım olabilir)
                ax.text(m_id, 0, "Idle", ha='center', va='center',
                       fontsize=9, color='gray', style='italic')
            else:
                # Makinede bir operatör var
                ax.text(m_id, 0, f"O{op_id}", ha='center', va='center',
                       fontsize=10, fontweight='bold', color='black')
        
        # Hangi operatörlerin makinelere atandığını ve becerilerini takip et
        operator_to_machine = {}
        operator_skill_on_machine = {}
        for m_id, op_id in enumerate(machine_assignments):
            if op_id != -1:
                operator_to_machine[op_id] = m_id
                # Bu operatörün bu makinedeki beceri skorunu al
                skill = operator_skills[m_id] if m_id < len(operator_skills) else 0.0
                operator_skill_on_machine[op_id] = skill
        
        # Operatörleri beceri bilgisiyle renkli daireler olarak çiz
        for op_id in range(num_operators):
            color = operator_colors[op_id]
            
            if op_id in operator_to_machine:
                # Operatör bir makinede çalışıyor
                m_id = operator_to_machine[op_id]
                x_pos = m_id
                y_pos = 0.6
                skill = operator_skill_on_machine.get(op_id, 0.0)
                
                # Operatör dairesini çiz (boyut beceriye göre - daha yüksek beceri = daha büyük daire)
                # Aktif işçiler biraz daha büyük ve daha kalın kenarlı
                circle_size = 0.14 + (skill * 0.08)  # Aralık: 0.14'ten 0.22'ye (0.12'den artırıldı)
                circle = plt.Circle((x_pos, y_pos), circle_size, facecolor=color, 
                                 edgecolor='black', linewidth=3, zorder=10, alpha=0.8)  # linewidth 2'den artırıldı
                ax.add_patch(circle)
                
                # Operatörü beceri skoruyla etiketle
                skill_text = f"O{op_id}\n{skill:.2f}" if skill >= 0 else f"O{op_id}"
                ax.text(x_pos, y_pos, skill_text, ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
                
                # Beceri skorunu operatörün altında göster
                ax.text(x_pos, y_pos - 0.25, f"skill:{skill:.2f}", ha='center', va='top',
                       fontsize=7, color='black', style='italic')
            else:
                # Operatör boş - park alanına koy
                parking_x = num_machines + (op_id * 0.4)
                x_pos = parking_x
                y_pos = 0.6
                
                # Operatör dairesini çiz (boşken daha küçük)
                circle = plt.Circle((x_pos, y_pos), 0.12, facecolor=color, 
                                 edgecolor='black', linewidth=1.5, zorder=10, alpha=0.5)
                ax.add_patch(circle)
                
                # Operatörü etiketle
                ax.text(x_pos, y_pos, f"O{op_id}", ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        
        # Park alanı etiketi çiz
        if num_operators > 0:
            parking_start = num_machines
            ax.text(parking_start + (num_operators - 1) * 0.2, 0.9, "Boş Alan",
                   ha='center', va='bottom', fontsize=9, style='italic', color='gray')
        
        # Çizim limitlerini ve etiketlerini ayarla
        x_max = num_machines + num_operators * 0.4
        ax.set_xlim(-0.5, x_max + 0.5)
        # Alt tarafta makine durum etiketlerine yer açmak için y-limitleri genişlet
        ax.set_ylim(-1.0, 1.2)
        
        # Daha temiz görünüm için spine'ları gizle
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Tick'leri kaldır
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Zaman, vardiya, episode bilgisiyle başlık ayarla
        title_text = f"{title}\n"
        if episode_number > 0:
            title_text += f"Episode {episode_number} | "
        title_text += f"t={current_time:.1f} dk | vardiya={shift_index} | iyi parça={produced_parts}"
        ax.set_title(title_text, fontsize=11, pad=10)
    
    # Animasyon oluştur
    anim = FuncAnimation(fig, update, frames=len(history), interval=1000/fps, repeat=True)
    
    # GIF olarak kaydet
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    
    print(f"GIF {output_path} dosyasına kaydedildi ({len(history)} frame)")
