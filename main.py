"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
# from final.env import ArmEnv
# from final.rl import DDPG

# from _3DOF_Pytorch_non_pl.env import ArmEnv
# # from rl import DDPG
# from _3DOF_Pytorch_non_pl.rl_torch import DDPG

from env import ArmEnv
# from rl import DDPG
from rl_torch import DDPG
import numpy as np

MAX_EPISODES = 900
MAX_EP_STEPS = 300
ON_TRAIN = 0  #True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # 是否载入原有模型
    # rl.restore()

    reward_all = []
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()
            # print(f"goal = {env.goal}")
            # print(f"xyz = {s[3],s[4],s[5]}")

            a = rl.choose_action(s)
            # print(f"a={a}")

            s_, r, done, _ = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))

                reward_all.append(ep_r)
                break
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    import os

    plt.figure(figsize=(10, 6))
    plt.ylabel('reward_all')
    plt.xlabel('training steps')
    plt.plot(np.arange(len(reward_all)), reward_all)
    # rl.save()  rl.save()
    # plt.show()

    # from datetime import datetime
    #
    # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = rl.save()
    file_name = f'params_{current_time}.png'  # 文件名
    save_path = './model_save'
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)

    # 存数据
    import pandas as pd

    # 创建 DataFrame
    df = pd.DataFrame({
        'len(reward_all)': len(reward_all),
        'reward_all': reward_all
    })
    file_name = f'params_{current_time}.xlsx'  # 文件名
    save_path = './model_save'
    file_path = os.path.join(save_path, file_name)
    # 保存 DataFrame 到 Excel 文件
    df.to_excel(file_path, index=False)
    print(f"save train result as {file_name}")
    plt.show()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    # print(f"s = {s}")
    env.set_goal(0, 42.5,39.23)
    timer = 0
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done, angle_all = env.step(a)
        # print(f"angle_all = {angle_all}")

        # timer +=1
        # if timer % 800 == 200:
        #     env.set_goal(100, 300)
        # if timer % 800 == 400:
        #     env.set_goal(100, 100)
        # if timer % 800 == 600:
        #     env.set_goal(300, 100)
        # if timer % 800 == 0:
        #     env.set_goal(300, 300)


def eval_p2p():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    # s = env.reset()
    s, r, done, angle_all = env.reset_start()
    print(f"s = {s}")
    env.set_goal(20,42.5,39.2)  #  修改
    done = 0
    done_4p = 0
    timer = 0
    traj_all = []
    traj_all.append((s[3] , s[4] , s[5] ))
    traj_q_all = []
    traj_q_all.append((angle_all[0], angle_all[1], angle_all[2]))

    ang_traj = []
    while not done_4p:
        env.render()
        a = rl.choose_action(s)
        s, r, done, angle_all = env.step(a)

        print(f"xyz = {s[3] , s[4] , s[5] }")
        traj_all.append((s[3] , s[4] , s[5] ))
        traj_q_all.append((angle_all[0], angle_all[1], angle_all[2]))
        print(f"angle_all = {angle_all}")
        print(f"r = {r}")

        timer += 1
        if timer > 300:
            done_4p = 1

    x_vals = [point[0] for point in traj_all]
    y_vals = [point[1] for point in traj_all]
    z_vals = [point[2] for point in traj_all]
    print(f"orig = {x_vals[0],y_vals[0] , z_vals[0]}")

    q1_vals = [point[0] for point in traj_q_all]
    q2_vals = [point[1] for point in traj_q_all]
    q3_vals = [point[2] for point in traj_q_all]

    # print(f"q1_vals = {q1_vals}")
    # print(f"q2_vals = {q2_vals}")
    # print(f"q2_vals = {q3_vals}")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a new figure for 3D plotting
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    # Plotting the 3D curve
    ax1.plot(x_vals, y_vals, z_vals, label='3D Curve')
    # ax1.scatter(0, 42.5, 39.23, c='red', marker='o', s=100) #self.goal['x'],self.goal['y'],self.goal['z']
    ax1.scatter(env.goal['x'],env.goal['y'],env.goal['z'], c='red', marker='o', s=100)
    ax1.scatter(81.73, 0, 0, c='green', marker='o', s=100)

    # Setting labels for the axes
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')

    # Display the legend
    ax1.legend()

    # Show the plot
    # plt.show()

    # 画出关节角度图像  Draw the joint Angle image
    # Create a new figure for 3D plotting
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    # Plotting the 3D curve
    ax2.plot(q1_vals, q2_vals, q3_vals, label='3D Curve')

    # Setting labels for the axes
    ax2.set_xlabel('q1 axis')
    ax2.set_ylabel('q2 axis')
    ax2.set_zlabel('q3 axis')

    # Display the legend
    ax2.legend()

    fig3 = plt.figure()#figsize=(10,7.5)
    ax3 = fig3.add_subplot(111) # 2D 图像

    time_steps = np.linspace(0, 1, len(q1_vals))

    # Plotting the 2D curve
    # ax3.figure(figsize=(10, 6))
    ax3.plot(time_steps, q1_vals[:], label='Joint 1')
    ax3.plot(time_steps, q2_vals[:], label='Joint 2')
    ax3.plot(time_steps, q3_vals[:], label='Joint 3')



    ax3.set_xlabel('Normalized Time', fontsize=14)
    ax3.set_ylabel('Joint Angle (radians)', fontsize=14)
    ax3.set_title('Joint Angles over Time', fontsize=16)
    ax3.grid(True)

    # Display the legend
    ax3.legend(fontsize=10)

    # Show the plot
    plt.show()


if ON_TRAIN:
    train()
else:
    # eval()
    eval_p2p()




