import mjx
from mjx.agents import ShantenAgent
import json
import numpy as np

agent = ShantenAgent()
env = mjx.MjxEnv()
obs_dict = env.reset()

obs_hist = []
action_hist = []

for j in range(100):
    while not env.done():
        actions = {}
        for player_id, obs in obs_dict.items():
            legal_actions = obs.legal_actions()
            action = agent.act(obs)
            actions[player_id] = action

            # 選択できるアクションが複数ある場合、obsとactionを保存する
            if len(legal_actions) > 1:
                obs_hist.append(obs.to_features(feature_name="mjx-small-v0").ravel())
                action_hist.append(action.to_idx())
        
        obs_dict = env.step(actions)
    env.reset()

# ファイルに書き出し
np.save("shanten_obs.npy", np.stack(obs_hist))
np.save("shanten_actions.npy", np.array(action_hist, dtype=np.int32))