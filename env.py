import numpy as np
import pyglet


class ArmEnv(object):
    viewer = None
    dt = .1    # refresh rate
    action_bound = [-1, 1]
    goal = {'x': 0., 'y': 42.5, 'z': 39.23,'l': 20}  # 40
    state_dim = 13
    action_dim = 3

    def __init__(self):
        self.arm_info = np.zeros(
            3, dtype=[('l', np.float32), ('r', np.float32)])
        # self.arm_info['l'] = 100        # 2 arms length  42.5  39.23
        self.arm_info['l'] = [8.95, 42.5 , 39.23]
        # self.arm_info['r'] = np.pi/6    # 2 angles information
        self.arm_info['r'] = [0.0, 0.0, 0.0]
        self.on_goal = 0

    def update_state_all(self):
        done = False
        (d0, a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a0r , a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xyz = np.array([0., 0., 0.])  # a1 start (x0, y0)
        a1xyz_ = np.array([np.cos(a1r)* a1l * np.cos(a0r)  , np.cos(a1r)* a1l * np.sin(a0r)   ,  np.sin(a1r)* a1l])  + a1xyz  # a1 end and a2 start (x1, y1 ,z1)
        finger = np.array([np.cos(a1r+a2r)* a2l * np.cos(a0r)  , np.cos(a1r+a2r)* a2l * np.sin(a0r)   ,  np.sin(a1r+a2r)* a2l])  + a1xyz_  # a2 end (x2, y2)



        # normalize features
        dist1 = [(self.goal['x'] - a1xyz_[0]) / 400, (self.goal['y'] - a1xyz_[1]) / 400, (self.goal['z'] - a1xyz_[2]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400, (self.goal['z'] - finger[2]) / 400]
        r = -np.sqrt(dist2[0] ** 2 + dist2[1] ** 2 +  dist2[2] ** 2)
        # print(f"dist2={dist2[0]*400} {dist2[1]*400} {dist2[2]*400}")
        on_goal_flag = 0

        # done and reward  if (finger[0] - self.goal['x'])**2 + (finger[1] - self.goal['y'])**2 < (self.goal['l'] / 2)**2:
        if (finger[0] - self.goal['x']) ** 2 + (finger[1] - self.goal['y']) ** 2 + (finger[2] - self.goal['z']) ** 2 < (self.goal['l'] / 2) ** 2:

            r += 1.
            self.on_goal += 1
            on_goal_flag = 1
            if self.on_goal > 50:
                done = True
        else:
            self.on_goal = 0
        # print(f"on_goal_flag = {on_goal_flag}")


        # state
        s = np.concatenate((a1xyz_, finger, dist1 , dist2, [1. if self.on_goal else 0.])) #/200? 是否需要
        return s, r, done, self.arm_info['r']

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        s, r, done, self.arm_info['r'] = self.update_state_all()
        return s, r, done, self.arm_info['r']

    def reset(self):   #  reset in train
        # self.goal['x'] = np.random.rand()*400.
        # self.goal['y'] = np.random.rand()*400.

        (_ , a1l, a2l) = self.arm_info['l']
        l_max = a1l + a2l
        # print(f"l_max = {l_max}")
        theta = np.random.uniform(0, 2 * np.pi)  # 随机生成角度 θ，范围 [0, 2π)
        phi = np.random.uniform(0, np.pi)  # 随机生成角度 φ，范围 [0, π)
        r = l_max * np.cbrt(np.random.uniform(0, 1))  # 随机生成半径 r，使用 cbrt(random) 保证均匀分布

        # self.goal['x'] = r * np.sin(phi) * np.cos(theta)  # 将球坐标转换为笛卡尔坐标 x
        # self.goal['y'] = r * np.sin(phi) * np.sin(theta)  # 将球坐标转换为笛卡尔坐标 y
        # self.goal['z'] = r * np.cos(phi)  # 将球坐标转换为笛卡尔坐标 z

        r_new = 30 * np.cbrt(np.random.uniform(0, 1))
        self.goal['x'] = 0  + r_new * np.sin(phi) * np.cos(theta)# 将球坐标转换为笛卡尔坐标 x
        self.goal['y'] = 42.5 + r_new * np.sin(phi) * np.sin(theta)# 将球坐标转换为笛卡尔坐标 y
        self.goal['z'] = 39.23 + r_new * np.cos(phi)# 将球坐标转换为笛卡尔坐标 z
        print(f"goal = {self.goal['x'],self.goal['y'],self.goal['z']}")

        # self.arm_info['r'] = 2 * np.pi * np.random.rand(3)
        self.arm_info['r'] = 0

        self.on_goal = 0

        s, r, done, self.arm_info['r'] = self.update_state_all()
        return s

    def reset_start(self):  # reset start point at ( 0 , 0)

        self.arm_info['r'] = np.array([0,0,0])
        self.on_goal = 0

        s, r, done, self.arm_info['r'] = self.update_state_all()
        return s, r, done, self.arm_info['r']

    def set_goal(self,goal_x ,goal_y,goal_z):
        self.goal['x'] = goal_x
        self.goal['y'] = goal_y
        self.goal['z'] = goal_z

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(3)-0.5    # 3 random radians


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        self.center_coord = np.array([200, 200])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x']+200 - goal['l'] / 2, goal['y']+200 - goal['l'] / 2,                # location
                     goal['x']+200 - goal['l'] / 2, goal['y']+200 + goal['l'] / 2,
                     goal['x']+200 + goal['l'] / 2, goal['y']+200 + goal['l'] / 2,
                     goal['x']+200 + goal['l'] / 2, goal['y']+200 - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            self.goal_info['x']+200 - self.goal_info['l']/2, self.goal_info['y']+200 - self.goal_info['l']/2,
            self.goal_info['x']+200 + self.goal_info['l']/2, self.goal_info['y']+200 - self.goal_info['l']/2,
            self.goal_info['x']+200 + self.goal_info['l']/2, self.goal_info['y']+200 + self.goal_info['l']/2,
            self.goal_info['x']+200 - self.goal_info['l']/2, self.goal_info['y']+200+ self.goal_info['l']/2)

        # update arm

        (d0, a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a0r, a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))

    # convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y



if __name__ == '__main__':
    env = ArmEnv()
    # while True:
    #     env.render()
    #     env.step(env.sample_action())
    traj_all = []
    traj_q_all = []
    env.reset_start()
    env.set_goal(40,40,40)
    for _ in range(4000):
        env.render()
        a = [0.00, 0.00, 0.01]
        s, r, done, angle_all = env.step(a)
        print(f"xyz = {s[3] , s[4] ,s[5]}")
        traj_all.append((s[3] , s[4] ,s[5]))
        traj_q_all.append((angle_all[0], angle_all[1], angle_all[2]))

    x_vals = [point[0] for point in traj_all]
    y_vals = [point[1] for point in traj_all]
    z_vals = [point[2] for point in traj_all]

    q1_vals = [point[0] for point in traj_q_all]
    q2_vals = [point[1] for point in traj_q_all]
    q3_vals = [point[2] for point in traj_q_all]

    print(f"q1_vals = {q1_vals}")
    print(f"q2_vals = {q2_vals}")
    print(f"q3_vals = {q3_vals}")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the 3D curve
    ax.plot(x_vals, y_vals, z_vals, label='3D Curve')

    # Setting labels for the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Display the legend
    ax.legend()

    # Show the plot
    plt.show()

    # 画出关节角度图像  Draw the joint Angle image
    # Create a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the 3D curve
    ax.plot(q1_vals, q2_vals, q3_vals, label='3D Curve')

    # Setting labels for the axes
    ax.set_xlabel('q1 axis')
    ax.set_ylabel('q2 axis')
    ax.set_zlabel('q3 axis')

    # Display the legend
    ax.legend()

    # Show the plot
    plt.show()

    # 创建三维图形对象
    # plt.figure()
    # # ax = fig.add_subplot(111, projection='2d')
    #
    # # 绘制三维曲线
    # # ax.plot(x_vals, y_vals,  label='3D Curve') #z,
    # q1_vals = [0 if x > 6.18 else x for x in q1_vals]
    # q2_vals = [0 if x > 6.18 else x for x in q2_vals]
    #
    # plt.plot(q1_vals, q2_vals)
    # # 设置标签
    # plt.xlabel('q1_vals ')
    # plt.ylabel('q2_vals ')
    # # ax.set_zlabel('Z 轴')
    #
    # # # 显示图例
    # plt.legend()
    # plt.show()