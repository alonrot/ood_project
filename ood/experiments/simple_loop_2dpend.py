#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
import numpy as np
import pybullet_envs
import time
import pdb
from policyNN_swingup import SmallReactivePolicy

import matplotlib.pyplot as plt

"""
### TODO
1) Code up the dynamics model: Give flexibility (inheritance) for features to be NNs or Sarkka features. Think about re-using this for the koopman project
2) Record trajectories from the swin-up policy and compute the model posterior
3) Code up a OOD function detetion using the likelihood (make a virtual class for the different possible implementations of the OOD function)
2) Create a bunch of new environments with different forces/stops, etc.
2) Achieve the trigger of OOD
"""



def simulate_single_pend(Nsteps,visualize=False,plot=False):
  
    # Load gym environment:
    env = gym.make("InvertedPendulumSwingupBulletEnv-v0")
    # pdb.set_trace()
    if visualize:
        env.render(mode="human")

    # NN Swing-up policy:
    pi = SmallReactivePolicy(env.observation_space, env.action_space)

    # while 1:
    # frame = 0
    # score = 0
    # restart_delay = 0
    ii = 0
    obs = env.reset()

    # Define observation vector:
    # pdb.set_trace()
    dim_obs = env.observation_space._shape[0] 
    obs_vec = np.zeros((Nsteps,dim_obs))
    rew_vec = np.zeros(Nsteps)
    u_vec = np.zeros(Nsteps)
    done = False

    while ii < Nsteps and not done:

        # obs: return np.array([x, vx, np.cos(self.theta), np.sin(self.theta), theta_dot])
        # reward = np.cos(self.robot.theta)
        
        time.sleep(1. / 60.)
        a = pi.act(obs)
        obs, r, done, _ = env.step(a)
        # score += r
        # frame += 1
        still_open = env.render("human")
        if still_open == False:
            return

        # Store data:
        obs_vec[ii,:] = obs[:]
        rew_vec[ii] = r
        u_vec[ii] = a

        ii += 1
        
        
        # if not done:
        #     continue
        
        # if restart_delay == 0:
        #     print("score=%0.2f in %i frames" % (score, frame))
        #     restart_delay = 60 * 2  # 2 sec at 60 fps
        # else:
        #     restart_delay -= 1
        
        #     if restart_delay > 0:
        #         continue
            
        #     break



    if done:
        raise NotImplementedError

    # Plotting:
    if plot:
        hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
        # hdl_fig.suptitle("Inverted pendulum simulation x(t)")
        hdl_splots[0].plot(np.arange(0,Nsteps),rew_vec,linestyle="None",marker=".",color="k")
        # hdl_splots[1].plot(np.arange(0,Nevals+1,1),x_traj[1,:],linestyle="None",marker=".",color="k")
        plt.show(block=True)


    # Return policy function:
    policy_fun = pi.act


    return obs_vec, u_vec, policy_fun

if __name__ == "__main__":
  
    obs_vec = simulate_single_pend(Nsteps=60*2)
