"""
Eğitilmiş Q-learning ajanı için değerlendirme script'i.

Bu script, eğitilmiş ajanı greedy politika ile test eder ve sonuçları gösterir.
"""

import numpy as np
from pathlib import Path

from config.demo_config import get_demo_config
from env.factory_env import FactoryEnv
from agent.q_learning_agent import QLearningAgent
from utils.visualization import render_timeline_gif


def run_greedy_episode(env: FactoryEnv, agent: QLearningAgent, record_history: bool = False):
    """
    Tamamen greedy politika ile bir episode çalıştırır (keşif yok).
    
    record_history True ise, env.reset(record_history=True) çağır.
    Aksi halde, env.reset() çağır.
    
    Greedy eylem seçimi için:
    - Verilen bir durum için, argmax_a Q(durum, a) seç.
    - Durum daha önce hiç görülmemişse, varsayılan olarak eylem 0'ı seç.
    
    Args:
        env: FactoryEnv örneği
        agent: QLearningAgent örneği
        record_history: True ise, görselleştirme için olay geçmişini kaydet
    
    Returns:
        (total_return, produced_good_parts) tuple'ı
    """
    if record_history:
        state = env.reset(record_history=True)
    else:
        state = env.reset()
    
    done = False
    total_return = 0.0
    
    while not done:
        # Tamamen greedy politika: en yüksek Q-değerine sahip eylemi seç
        # Durum görülmemişse, varsayılan olarak eylem 0
        if state in agent.Q:
            q_values = {a: agent.Q[state].get(a, 0.0) for a in range(agent.num_actions)}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            # Aynı Q-değerine sahip birden fazla eylem varsa, ilkini seç
            action = best_actions[0]
        else:
            # Durum daha önce hiç görülmemiş, varsayılan olarak eylem 0
            action = 0
        
        next_state, reward, done, info = env.step(action)
        total_return += reward
        state = next_state
    
    return total_return, env.produced_good_parts


def main():
    """Ana değerlendirme fonksiyonu."""
    # Yapılandırmayı yükle
    config = get_demo_config()
    
    # Ortamı oluştur
    env = FactoryEnv(config, seed=123)  # Değerlendirme için farklı tohum
    
    # Eylem sayısını belirle
    num_actions = config["num_operators"] + 1
    
    # Ajanı oluştur
    agent = QLearningAgent(num_actions=num_actions)
    
    # Eğitilmiş Q-tablosunu HDF5 dosyasından yükle
    q_table_path = Path("q_table.h5")
    if not q_table_path.exists():
        print(f"Hata: Q-tablosu dosyası {q_table_path} bulunamadı!")
        print("Lütfen önce eğitimi çalıştırın: py -m scripts.main_train")
        return
    
    agent.load_h5(str(q_table_path))
    print(f"Q-tablosu {q_table_path} dosyasından yüklendi")
    print(f"Q-tablosundaki durum-eylem çifti sayısı: {sum(len(actions) for actions in agent.Q.values())}")
    print("-" * 50)
    
    # Değerlendirme episode'larını çalıştır
    num_eval_episodes = 100
    print(f"{num_eval_episodes} değerlendirme episode'ı çalıştırılıyor (greedy politika)...")
    
    eval_returns = []
    eval_productions = []
    history = None  # İlk episode'dan history'yi sakla
    
    for episode in range(num_eval_episodes):
        # Sadece ilk episode için history kaydet
        if episode == 0:
            total_return, produced_parts = run_greedy_episode(env, agent, record_history=True)
            history = env.get_history()  # İlk episode'dan hemen sonra history'yi al
        else:
            total_return, produced_parts = run_greedy_episode(env, agent, record_history=False)
        
        eval_returns.append(total_return)
        eval_productions.append(produced_parts)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d}: return={total_return:7.2f}, üretilen={produced_parts:3d}")
    
    print("-" * 50)
    print("Değerlendirme Özeti:")
    print(f"  Ortalama return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")
    print(f"  Ortalama üretilen iyi parça: {np.mean(eval_productions):.1f} ± {np.std(eval_productions):.1f}")
    print(f"  Min üretilen: {np.min(eval_productions)}")
    print(f"  Max üretilen: {np.max(eval_productions)}")
    print(f"  Hedef üretim: {config['target_production_per_day']}")
    
    # Hedefin karşılanıp karşılanmadığını kontrol et
    target_met = sum(1 for p in eval_productions if p >= config['target_production_per_day'])
    print(f"  Hedefi karşılayan episode'lar: {target_met}/{num_eval_episodes} ({100*target_met/num_eval_episodes:.1f}%)")
    
    # İlk episode history'sinden değerlendirme GIF'i oluştur
    if history:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        render_timeline_gif(
            history=history,
            config=config,
            output_path=str(output_dir / "evaluation_run.gif"),
            title="Fabrika Operatör Ataması - Değerlendirme (Senaryo 1)",
            fps=10,
        )
        print(f"Değerlendirme GIF'i {output_dir / 'evaluation_run.gif'} dosyasına kaydedildi")
    else:
        print("Uyarı: GIF üretimi için kaydedilmiş history yok")


if __name__ == "__main__":
    main()
