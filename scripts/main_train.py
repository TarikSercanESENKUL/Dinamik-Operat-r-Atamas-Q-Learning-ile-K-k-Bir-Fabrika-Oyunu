"""
Fabrika ortamında Q-learning eğitimi için kullandığım basit script.

Kısaca: Ortamı kurup ajanı başlatıyorum, belirli sayıda episode koşturup
Q-tablosunu ve bazı grafik / GIF çıktıları kaydediyorum.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config.demo_config import get_demo_config
from env.factory_env import FactoryEnv
from agent.q_learning_agent import QLearningAgent
from utils.plotting import plot_training_curves
from utils.visualization import render_timeline_gif


def main():
    """Ana eğitim fonksiyonu (train)."""
    # Yapılandırmayı yükle
    config = get_demo_config()
    
    # Ortamı oluştur
    env = FactoryEnv(config, seed=42)
    
    # Eylem sayısını belirle
    # Eylemler: operatör 0, 1, 2, ..., num_operators-1 atama veya boş bırak (num_operators)
    num_actions = config["num_operators"] + 1
    
    # Eğitim parametreleri
    # 2000 episode'dan 10000 episode'a çıkarıldı (daha uzun ve kararlı eğitim için)
    num_episodes = 10000
    
    # Ajanı oluştur
    # Learning rate artık dinamik olarak azalacak; burada başlangıç değerini veriyoruz.
    agent = QLearningAgent(
        num_actions=num_actions,
        learning_rate=0.1,  # Başlangıç için daha yüksek öğrenme oranı (sonra kademeli olarak düşecek)
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=num_episodes - 1  # Son episode'a kadar doğrusal azalma
    )
    
    # GIF için history kaydedilecek episode'lar (her 100. episode, 100'den başlayarak)
    # Sadece bu episode'lar arasından "en iyi" olan için GIF üreteceğiz
    history_episode_indices = list(range(99, num_episodes, 100))  # 99 = episode 100 (0-indeksli)
    
    # En iyi episode'un history'sini saklamak için değişkenler
    best_episode_index = None
    best_episode_return = None
    best_episode_history = None
    
    # Takip listeleri
    episode_returns = []
    episode_productions = []
    
    print("Eğitim başlatılıyor...")
    print(f"Episode sayısı: {num_episodes}")
    print(f"Eylem sayısı: {num_actions}")
    print(f"History kaydedilecek episode'lar: {[i+1 for i in history_episode_indices[:5]]}... (her 100. episode)")
    print("-" * 50)
    
    # Eğitim döngüsü
    for episode in range(num_episodes):
        # GIF üretimi için her 100. episode'da history kaydet (episode 100'den başlayarak)
        if episode in history_episode_indices:
            state = env.reset(record_history=True)
        else:
            state = env.reset(record_history=False)
        
        done = False
        episode_return = 0.0
        
        while not done:
            action = agent.select_action(state, episode)
            next_state, reward, done, info = env.step(action)
            # Episode indeksini de vererek dinamik learning rate kullanımını etkinleştir
            agent.update(state, action, reward, next_state, done, episode_index=episode)
            episode_return += reward
            state = next_state
        
        # Episode istatistiklerini kaydet
        episode_returns.append(episode_return)
        episode_productions.append(env.produced_good_parts)
        
        # Kaydedilen episode'ların history'sini al ve en iyi episode'u takip et
        if episode in history_episode_indices:
            history = env.get_history()
            if history:
                # Her frame'e tanımlama için episode numarası ekle
                for frame in history:
                    frame["episode_number"] = episode + 1
                
                # Şu ana kadarki en iyi episode'u güncelle
                if best_episode_return is None or episode_return > best_episode_return:
                    best_episode_return = episode_return
                    best_episode_index = episode + 1  # 1-indeksli
                    best_episode_history = history
                
                print(f"Episode {episode + 1} için history kaydedildi ({len(history)} frame)")
        
        # İlerlemeyi yazdır (log gürültüsünü azaltmak için her 100 episode'da bir)
        if (episode + 1) % 100 == 0 or episode == 0:
            avg_return = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else episode_return
            avg_production = np.mean(episode_productions[-100:]) if len(episode_productions) >= 100 else env.produced_good_parts
            epsilon = agent.get_epsilon(episode)
            print(f"Episode {episode + 1:4d}: "
                  f"return={episode_return:7.2f}, "
                  f"üretilen={env.produced_good_parts:3d}, "
                  f"ort_return={avg_return:7.2f}, "
                  f"ort_üretim={avg_production:5.1f}, "
                  f"epsilon={epsilon:.3f}")
    
    print("-" * 50)
    print("Eğitim tamamlandı!")
    print(f"Son ortalama return (son 100 episode): {np.mean(episode_returns[-100:]):.2f}")
    print(f"Son ortalama üretim (son 100 episode): {np.mean(episode_productions[-100:]):.1f}")
    
    # Q-tablosunu her iki formatta kaydet
    q_table_pkl_path = "q_table.pkl"
    q_table_h5_path = "q_table.h5"
    agent.save(q_table_pkl_path)
    agent.save_h5(q_table_h5_path)
    print(f"Q-tablosu {q_table_pkl_path} dosyasına kaydedildi (pickle formatı)")
    print(f"Q-tablosu {q_table_h5_path} dosyasına kaydedildi (HDF5 formatı)")
    
    # Eğitim eğrilerini çiz
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    plot_training_curves(episode_returns, episode_productions, str(output_dir))
    print(f"Grafikler {output_dir}/ dizinine kaydedildi")
    
    # En iyi episode için tek bir eğitim GIF'i üret
    if best_episode_history is not None:
        print(f"\nEn iyi episode: {best_episode_index} (return={best_episode_return:.2f})")
        print("En iyi episode history'sinden eğitim GIF'i üretiliyor...")
        render_timeline_gif(
            history=best_episode_history,
            config=config,
            output_path=str(output_dir / "training_best_episode.gif"),
            title=f"Fabrika Operatör Ataması - En İyi Episode (#{best_episode_index})",
            fps=10,
        )
        print(f"Eğitim GIF'i {output_dir / 'training_best_episode.gif'} dosyasına kaydedildi")


if __name__ == "__main__":
    main()
