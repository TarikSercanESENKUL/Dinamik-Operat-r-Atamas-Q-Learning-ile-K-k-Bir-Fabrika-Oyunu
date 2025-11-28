"""Script to create a demo GIF from trained Q-table."""

from pathlib import Path
from config.demo_config import get_demo_config
from env.factory_env import FactoryEnv
from agent.q_learning_agent import QLearningAgent
from utils.visualization import render_timeline_gif


def main():
    """Create a demo GIF."""
    config = get_demo_config()
    env = FactoryEnv(config, seed=42)
    agent = QLearningAgent(num_actions=config["num_operators"] + 1)
    
    # Load trained Q-table
    agent.load_h5("q_table.h5")
    print("Loaded Q-table")
    
    # Run one episode with history
    state = env.reset(record_history=True)
    done = False
    
    while not done:
        action = agent.select_action(state, 0, use_greedy=True)
        state, reward, done, info = env.step(action)
    
    history = env.get_history()
    print(f"Recorded {len(history)} frames")
    
    # Create GIF
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    render_timeline_gif(
        history=history,
        config=config,
        output_path=str(output_dir / "final_demo.gif"),
        title="Factory - Final Demo",
        fps=10,
    )
    print("GIF created!")


if __name__ == "__main__":
    main()

