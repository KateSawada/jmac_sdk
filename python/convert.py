from mjx import Observation, State, Action 
import glob
import numpy as np
count = 0
files = glob.glob("./json/json/*")
obs_hist = []
action_hist = []
for file in files:
  print("#" + str(count) + "Loading file....")
  with open(file) as f:
    lines = f.readlines()
    for line in lines:
      state = State(line)
      for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
        obs = Observation._from_cpp_obj(cpp_obs)
        feature = obs.to_features(feature_name="mjx-small-v0")
        action = Action._from_cpp_obj(cpp_act)
        action_idx = action.to_idx()
        obs_hist.append(feature)
        action_hist.append(action_idx)
  count += 1
  if count >= 1000:
    break


np.save("obs.npy", np.stack(obs_hist))
np.save("actions.npy", np.array(action_hist, dtype=np.int32))