"""
Basit ama net bir test script'i.

Eğitilmiş Q-learning ajanını (`q_table.h5`) kullanarak birden fazla senaryoda
greedy politika ile test eder ve özet istatistikler ile bir adet GIF üretir.

Bu dosya, eğitimden sonra hızlıca "öğrendi mi?" sorusuna cevap vermek için tasarlandı.
"""

import numpy as np
from pathlib import Path
from typing import Dict

from config.demo_config import get_demo_config
from env.factory_env import FactoryEnv
from agent.q_learning_agent import QLearningAgent
from utils.visualization import render_timeline_gif


def run_greedy_episode(env: FactoryEnv, agent: QLearningAgent, record_history: bool = False) -> Dict:
    """
    Tamamen greedy politika ile tek bir episode çalıştır.

    Args:
        env: FactoryEnv örneği
        agent: Eğitilmiş QLearningAgent
        record_history: True ise env.reset(record_history=True) çağrılır ve history tutulur.

    Returns:
        Episode sonucunu içeren bir sözlük.
    """
    state = env.reset(record_history=record_history)

    done = False
    total_return = 0.0
    step_count = 0
    max_steps = 10000  # Güvenlik için üst sınır

    while not done and step_count < max_steps:
        # Durum daha önce görülmüşse en yüksek Q-değerine sahip eylemi seç
        if state in agent.Q and agent.Q[state]:
            q_values = agent.Q[state]
            # Sadece tanımlı eylemler üzerinden argmax al
            best_action = max(q_values, key=q_values.get)
            action = best_action
        else:
            # Durum hiç görülmemişse veya Q-tablosunda bu durum için hiç eylem yoksa,
            # varsayılan olarak eylem 0'ı seç (en sade fallback).
            action = 0

        next_state, reward, done, info = env.step(action)
        total_return += reward
        state = next_state
        step_count += 1

    return {
        "total_return": total_return,
        "produced_parts": env.produced_good_parts,
        "step_count": step_count,
        "final_time": env.current_time_minutes,
    }


def main() -> None:
    """Ana test fonksiyonu (greedy politika ile hızlı test)."""
    config = get_demo_config()
    env = FactoryEnv(config, seed=123)  # Test için sabit seed

    num_actions = config["num_operators"] + 1
    agent = QLearningAgent(num_actions=num_actions)

    # Q-tablosunu .h5 dosyasından yükle
    q_table_path = Path("q_table.h5")
    if not q_table_path.exists():
        print(f"Hata: Q-tablosu dosyası {q_table_path} bulunamadı!")
        print("Lütfen önce eğitimi çalıştırın: py -m scripts.main_train")
        return

    agent.load_h5(str(q_table_path))
    print(f"Q-tablosu {q_table_path} dosyasından yüklendi")
    print(f"Q-tablosundaki durum-eylem çifti sayısı: {sum(len(a) for a in agent.Q.values())}")
    print("-" * 70)

    # Test parametreleri
    num_test_episodes = 100
    print(f"{num_test_episodes} test episode'ı çalıştırılıyor (greedy politika)...\n")

    results = []
    history = None

    for episode in range(num_test_episodes):
        # Sadece ilk episode için history kaydı aç (GIF için)
        record_history = (episode == 0)
        result = run_greedy_episode(env, agent, record_history=record_history)
        results.append(result)

        if record_history:
            history = env.get_history()

        if (episode + 1) % 10 == 0:
            avg_last = np.mean([r["produced_parts"] for r in results[-10:]])
            print(f"Episode {episode + 1:3d}: son 10 bölüm ort. üretim = {avg_last:5.1f}")

    # Özet istatistikler
    print("\n" + "=" * 70)
    print("TEST ÖZETİ")
    print("=" * 70)

    returns = np.array([r["total_return"] for r in results])
    productions = np.array([r["produced_parts"] for r in results])

    print(f"\n📊 GENEL İSTATİSTİKLER:")
    print(f"  Episode sayısı           : {num_test_episodes}")
    print(f"  Hedef üretim (config)    : {config['target_production_per_day']} parça/gün")

    print(f"\n💰 RETURN:")
    print(f"  Ortalama                 : {returns.mean():.2f} ± {returns.std():.2f}")
    print(f"  Min / Medyan / Max       : {returns.min():.2f} / {np.median(returns):.2f} / {returns.max():.2f}")

    print(f"\n📦 ÜRETİM:")
    print(f"  Ortalama                 : {productions.mean():.1f} ± {productions.std():.1f}")
    print(f"  Min / Medyan / Max       : {productions.min()} / {np.median(productions):.1f} / {productions.max()}")

    target = config["target_production_per_day"]
    target_met = np.sum(productions >= target)
    print(f"  Hedefi karşılayan bölümler: {target_met}/{num_test_episodes} ({100*target_met/num_test_episodes:.1f}%)")

    # Basit performans kategorileri
    excellent = np.sum(productions >= 1.2 * target)
    good = np.sum((productions >= target) & (productions < 1.2 * target))
    acceptable = np.sum((productions >= 0.8 * target) & (productions < target))
    poor = np.sum(productions < 0.8 * target)

    print(f"\n📈 PERFORMANS KATEGORİLERİ:")
    print(f"  Mükemmel (≥120% hedef)   : {excellent} episode")
    print(f"  İyi (100-120% hedef)     : {good} episode")
    print(f"  Kabul edilebilir (80-100%): {acceptable} episode")
    print(f"  Zayıf (<80% hedef)       : {poor} episode")

    # GIF oluştur (sadece ilk episode'dan)
    if history:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # History çok uzunsa hafifçe örnekle (performans için)
        max_frames = 400
        if len(history) > max_frames:
            step = max(1, len(history) // max_frames)
            history = history[::step]
            print(f"\nHistory örneklenerek {len(history)} frame'e düşürüldü (GIF için).")

        print("\nGIF oluşturuluyor...")
        render_timeline_gif(
            history=history,
            config=config,
            output_path=str(output_dir / "test_run.gif"),
            title="Fabrika Operatör Ataması - Test (Episode 1, greedy)",
            fps=10,
        )
        print(f"🎬 Test GIF'i {output_dir / 'test_run.gif'} dosyasına kaydedildi")

    print("\n" + "=" * 70)
    print("Test tamamlandı.")


if __name__ == "__main__":
    main()

