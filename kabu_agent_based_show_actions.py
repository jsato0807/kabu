import numpy as np
import matplotlib.pyplot as plt

def plot_actions_and_price(current_price, agent_actions):
    """
    エージェントのアクションの時間変化とcurrent_priceの推移を可視化
    
    Parameters:
    - current_price: 価格の時系列データ（1D array, shape: [T]）
    - agent_actions: 各エージェントのアクションの履歴（2D array, shape: [num_agents, T]）
                     各エージェントのアクションは {1: ロング, -1: ショート, 0: ノーポジション}
    """

    num_agents, T = agent_actions.shape  # エージェント数, 時系列の長さ

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # (1) current_price のプロット
    axs[0].plot(current_price, label="Current Price", color="black")
    axs[0].set_ylabel("Current Price")
    axs[0].set_title("Current Price Over Time")
    axs[0].legend()
    axs[0].grid(True)

    # (2) エージェントのアクションの可視化（ヒートマップ）
    im = axs[1].imshow(agent_actions, aspect='auto', cmap="coolwarm", interpolation="nearest", extent=[0, T, 0, num_agents])
    axs[1].set_ylabel("Agent Index")
    axs[1].set_xlabel("Time")
    axs[1].set_title("Agent Actions Over Time")
    cbar = fig.colorbar(im, ax=axs[1], orientation="vertical")
    cbar.set_label("Action (-1: Sell, 0: No Position, 1: Buy)")

    plt.tight_layout()
    plt.show()
