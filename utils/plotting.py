"""
Eğitim görselleştirmesi için çizim yardımcıları.

Bu modül, eğitim sırasında episode return'leri ve üretim sayılarını
görselleştirmek için yardımcı fonksiyonlar içerir.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def plot_training_curves(episode_returns: List[float], episode_productions: List[int], output_dir: str) -> None:
    """
    Episode return'leri ve üretimleri için eğitim eğrilerini çizer.
    
    Args:
        episode_returns: Episode başına toplam return listesi
        episode_productions: Episode başına üretilen iyi parça listesi
        output_dir: Grafikleri kaydetmek için dizin
    """
    # Çıktı dizini yoksa oluştur
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Episode return'lerini çiz
    plt.figure(figsize=(10, 6))
    # Ham return eğrisini artık çizmiyoruz (çok gürültülü olduğu için); sadece yumuşatılmış eğriyi gösteriyoruz.
    
    # Hareketli ortalama ekle (gürültüyü azaltmak için daha büyük window)
    window_size = 200  # 100'den 200'e çıkarıldı → daha pürüzsüz bir plato
    if len(episode_returns) >= window_size:
        moving_avg = []
        for i in range(len(episode_returns)):
            start = max(0, i - window_size + 1)
            moving_avg.append(sum(episode_returns[start:i+1]) / (i - start + 1))
        plt.plot(moving_avg, linewidth=2.5, label=f'Hareketli Ortalama ({window_size} episode)', color='red')
    else:
        # Episode sayısı azsa, ham veriyi tek başına çiz (erken testler için)
        plt.plot(episode_returns, alpha=0.4, linewidth=0.8, label='Episode Return', color='lightblue')
    
    plt.xlabel('Episode')
    plt.ylabel('Toplam Return')
    plt.title('Eğitim: Zaman İçinde Episode Return\'leri')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "returns.png", dpi=150)
    plt.close()
    
    # Episode üretimlerini çiz
    plt.figure(figsize=(10, 6))
    # Ham üretim eğrisi yerine ağırlıklı olarak hareketli ortalamayı göster
    if len(episode_productions) >= window_size:
        moving_avg = []
        for i in range(len(episode_productions)):
            start = max(0, i - window_size + 1)
            moving_avg.append(sum(episode_productions[start:i+1]) / (i - start + 1))
        plt.plot(moving_avg, linewidth=2.5, label=f'Hareketli Ortalama ({window_size} episode)', color='red')
    else:
        plt.plot(episode_productions, alpha=0.4, linewidth=0.8, label='Üretilen İyi Parça', color='lightgreen')
    
    # Hedef üretim çizgisi ekle
    if len(episode_productions) > 0:
        # Config'den hedefi al (varsayılan 90)
        target = 90  # Config'den alınabilir
        plt.axhline(y=target, color='green', linestyle='--', linewidth=2, label=f'Hedef Üretim ({target})')
    
    plt.xlabel('Episode')
    plt.ylabel('Üretilen İyi Parça')
    plt.title('Eğitim: Zaman İçinde Üretilen İyi Parçalar')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "productions.png", dpi=150)
    plt.close()
    
    print(f"Grafikler {output_path}/ dizinine kaydedildi")
