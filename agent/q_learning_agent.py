"""
Basit tabular Q-learning ajanı.

Bu dosyada, operatör atama problemi için kullandığım Q-learning ajanı var.
Çok akademik yazmak yerine, ders projesi seviyesinde, anlaşılır tutmaya çalıştım.
"""

import pickle
import h5py
import numpy as np
import ast
from typing import Tuple, Dict, Optional


class QLearningAgent:
    """
    Epsilon-greedy keşif kullanan tabular Q-learning ajanı.

    Kısaca: (state, action) → Q değeri tutan bir tablo var ve bu tabloyu
    her adımda güncelliyoruz. Eylem seçiminde de klasik epsilon-greedy kullanıyorum.
    """
    
    def __init__(
        self,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 500,
    ):
        """
        Q-learning ajanını başlatıyorum.

        num_actions: Ortamda kaç farklı aksiyon var (genelde operatör sayısı + 1).
        Diğer parametreler de klasik RL sembollerine karşılık geliyor (alpha, gamma, epsilon).
        Burada değerleri teoriden çok, birkaç deneme-sonuç ile ayarladım.
        """
        self.num_actions = num_actions
        # Öğrenme oranı için başlangıç ve bitiş değerleri
        # learning_rate: başlangıç alfa (erken hızlı öğrenme)
        self.learning_rate_start = learning_rate
        # Sonlara doğru daha küçük bir alfa ile daha stabil plato için
        self.learning_rate_end = learning_rate * 0.1
        self.learning_rate = learning_rate  # Geriye dönük uyumluluk için (güncel alfa değeri tutulur)
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        
        # Q-tablosu: (durum_tuple, eylem_int) -> float eşlemesi
        self.Q: Dict[Tuple, Dict[int, float]] = {}
    
    def get_epsilon(self, episode_index: int) -> float:
        """
        Verilen episode için o anda kullanacağımız epsilon değerini döndürür.

        Burada iki aşamalı basit bir program var:
        - İlk kısımda epsilon hızlıca düşüyor (ajan ortamı kaba taslak tanıyor).
        - Sonrasında daha yavaş düşüyor (öğrendiklerini ince ayar yapıyor).
        """
        if episode_index >= self.epsilon_decay_episodes:
            return self.epsilon_end

        # Toplam decay aralığını iki parçaya böl
        split_point = int(0.3 * self.epsilon_decay_episodes)
        if split_point <= 0:
            split_point = 1

        if episode_index <= split_point:
            # Hızlı ilk faz: 1.0 -> yaklaşık 0.3
            start = self.epsilon_start
            mid = max(0.3, self.epsilon_end)
            ratio = episode_index / split_point
            epsilon = start + ratio * (mid - start)
        else:
            # Yavaş ikinci faz: 0.3 -> epsilon_end
            remaining = self.epsilon_decay_episodes - split_point
            if remaining <= 0:
                return max(self.epsilon_end, 0.0)
            ratio = (episode_index - split_point) / remaining
            start = max(0.3, self.epsilon_end)
            end = self.epsilon_end
            epsilon = start + ratio * (end - start)

        return max(min(epsilon, self.epsilon_start), self.epsilon_end)

    def get_learning_rate(self, episode_index: int) -> float:
        """
        Hangi episode'da olduğumuza göre öğrenme oranını (alpha) ayarlıyorum.

        Başlarda alfa daha büyük (hızlı değişsin), sonlara doğru daha küçük
        (öğrendiklerini fazla bozmasın, grafik biraz daha düz olsun).
        """
        if episode_index >= self.epsilon_decay_episodes:
            alpha = self.learning_rate_end
        else:
            # Basit doğrusal azalma: learning_rate_start -> learning_rate_end
            ratio = episode_index / max(1, self.epsilon_decay_episodes)
            alpha = self.learning_rate_start + ratio * (self.learning_rate_end - self.learning_rate_start)

        # İçeride de güncel alfa değerini saklayalım
        self.learning_rate = float(alpha)
        return self.learning_rate
    
    def select_action(self, state: Tuple, episode_index: int, use_greedy: bool = False) -> int:
        """
        Verilen state için hangi aksiyonu seçeceğimize karar veriyoruz.

        Eğitimde epsilon-greedy, değerlendirmede ise tamamen greedy kullanıyorum.
        """
        import random
        
        if use_greedy:
            epsilon = 0.0  # Değerlendirmede keşif yok
        else:
            epsilon = self.get_epsilon(episode_index)
        
        # Epsilon-greedy: epsilon olasılığıyla keşif yap
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Greedy: en yüksek Q-değerine sahip eylemi seç
        q_values = {}
        for action in range(self.num_actions):
            q_values[action] = self.Q.get(state, {}).get(action, 0.0)
        
        # En yüksek Q-değerini bul
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        # Aynı Q-değerine sahip birden fazla eylem varsa, aralarından rastgele seç
        return random.choice(best_actions)
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool,
        episode_index: Optional[int] = None,
    ):
        """
        Klasik Q-learning güncellemesini yapıyorum:
        Q(s,a) ← Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') − Q(s,a))

        Not: Burada formülü bire bir uygulayıp, gereksiz ek karmaşıklık koymadım.
        """
        # Q-değerleri yoksa başlat
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = 0.0
        
        # Mevcut Q-değerini al
        current_q = self.Q[state][action]
        
        # Hedef Q-değerini hesapla
        if done:
            target_q = reward  # Episode bitti, gelecek ödül yok
        else:
            # Bir sonraki durum için maksimum Q-değerini bul
            if next_state in self.Q:
                max_next_q = max(
                    self.Q[next_state].get(a, 0.0)
                    for a in range(self.num_actions)
                )
            else:
                # Bir sonraki durum daha önce görülmemiş, varsayılan 0
                max_next_q = 0.0
            
            target_q = reward + self.discount_factor * max_next_q
        
        # Bu adım için öğrenme oranını belirle
        alpha = self.learning_rate if episode_index is None else self.get_learning_rate(episode_index)

        # Q-learning güncellemesi
        self.Q[state][action] = current_q + alpha * (target_q - current_q)
    
    def save(self, path: str):
        """
        Q-tablosunu pickle kullanarak diske kaydeder.
        
        Args:
            path: Q-tablosunu kaydetmek için dosya yolu
        """
        with open(path, 'wb') as f:
            pickle.dump(self.Q, f)
    
    def load(self, path: str):
        """
        Q-tablosunu pickle kullanarak diskten yükler.
        
        Args:
            path: Q-tablosunu yüklemek için dosya yolu
        """
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)
    
    def save_h5(self, path: str) -> None:
        """
        Q-tablosunu HDF5 (.h5) formatında kaydeder.
        
        Q-tablosu, verimli çapraz dil erişimi için matris formunda saklanır:
        - "state_keys": durum tuple'larını temsil eden UTF-8 kodlu string dizisi
        - "q_values": (num_states, num_actions) şeklinde 2D float32 dizisi
        
        Bu format, Q-tablolarını farklı programlama dilleri arasında paylaşmak ve
        büyük Q-tablolarını verimli şekilde saklamak için uygundur. Not: Bu hala
        tabular Q-learning (sinir ağı değil); sadece kalıcılık için HDF5 kullanıyoruz.
        
        Args:
            path: Q-tablosunu kaydetmek için dosya yolu
        """
        # Tüm benzersiz durumları topla
        state_to_index = {}
        for state, _ in self.Q.items():
            if state not in state_to_index:
                state_to_index[state] = len(state_to_index)
        
        num_states = len(state_to_index)
        
        # Q-değeri matrisini ayır
        q_values = np.zeros((num_states, self.num_actions), dtype=np.float32)
        
        # Q-değerlerini doldur
        for state, action_dict in self.Q.items():
            s_idx = state_to_index[state]
            for action, value in action_dict.items():
                q_values[s_idx, action] = value
        
        # Durum anahtarlarını string olarak oluştur (tutarlılık için indekse göre sıralı)
        sorted_states = sorted(state_to_index.items(), key=lambda kv: kv[1])
        state_keys = [repr(state) for state, _ in sorted_states]
        
        # HDF5'e kaydet
        with h5py.File(path, 'w') as f:
            # Durum anahtarlarını sabit uzunluklu string olarak sakla (S256 en fazla 256 karakter)
            f.create_dataset("state_keys", data=np.array(state_keys, dtype="S256"))
            f.create_dataset("q_values", data=q_values)
    
    def load_h5(self, path: str) -> None:
        """
        Daha önce save_h5() ile kaydedilmiş bir HDF5 (.h5) dosyasından Q-tablosunu yükler.
        
        Args:
            path: Q-tablosunu yüklemek için dosya yolu
        """
        with h5py.File(path, 'r') as f:
            # Durum anahtarlarını oku ve byte'lardan string'e çöz
            state_keys_bytes = f["state_keys"][:]
            state_keys = [key.decode('utf-8') for key in state_keys_bytes]
            
            # Q-değerleri matrisini oku
            q_values = f["q_values"][:]
        
        # Q-tablosu dictionary'sini yeniden oluştur
        self.Q = {}
        for i, state_str in enumerate(state_keys):
            # Durum tuple string'ini güvenli şekilde değerlendirmek için ast.literal_eval kullan
            state = ast.literal_eval(state_str)
            self.Q[state] = {}
            for action in range(self.num_actions):
                q_val = float(q_values[i, action])
                if q_val != 0.0:  # Bellek tasarrufu için sadece sıfır olmayan Q-değerlerini sakla
                    self.Q[state][action] = q_val

