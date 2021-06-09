"""
2020年11月25日 FIXED 禁止修改
用于 books\cct\CCT几何分析并解决rib宽度问题.md

增加坐标 x y z

gif 压缩：https://gifcompressor.com/zh/
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.animation as animation

fig = plt.figure(figsize=(11, 4.8))
# 绝对坐标系，确定绘图板位置
ax = plt.axes([0.15, 0.10, 0.35, 0.80])
ax3 = plt.axes([0.5, -0.05, 0.6, 1.3], projection="3d")

# 圆柱半径
r = 1.0
# 圆柱长度
length = 3.0
# 圆柱圆周方向分段
ksi_number = 500
# 圆柱轴向分段 注意 Y 方向是圆柱轴向
z_number = 500

# CCT
number = 720
theta_steps = np.linspace(0, 2 * 2 * np.pi, number)
z0 = 0.0
start_z = 1.0
c0 = 0.0
c1 = -0.3
# 二维螺线
x2 = theta_steps
y2 = (
    z0 / (2 * np.pi) * theta_steps
    + start_z
    + c0 * np.sin(theta_steps)
    + c1 * np.sin(2 * theta_steps)
)
# 三维螺线
x3 = r * np.sin(theta_steps)
z3 = r * np.cos(theta_steps)
y3 = y2


# 动画绘制
if True:
    an1 = ax.plot(x2, y2, "r-")[0]
    an31 = ax3.plot(x3[0 : 90 + 40], y3[0 : 90 + 40], z3[0 : 90 + 40], "r-")[0]
    an32 = ax3.plot(x3[90:320], y3[90:320], z3[90:320], "r--")[0]
    an33 = ax3.plot(x3[320 : 450 + 40], y3[320 : 450 + 40], z3[320 : 450 + 40], "r-")[0]
    an34 = ax3.plot(x3[450:660], y3[450:660], z3[450:660], "r--")[0]
    an35 = ax3.plot(x3[660:], y3[660:], z3[660:], "r-")[0]

    def update_2d(c1):
        y2 = (
            z0 / (2 * np.pi) * theta_steps
            + start_z
            + c0 * np.sin(theta_steps)
            + c1 * np.sin(2 * theta_steps)
        )

        an1.set_data(x2, y2)

    def update_3d(c1):
        y2 = (
            z0 / (2 * np.pi) * theta_steps
            + start_z
            + c0 * np.sin(theta_steps)
            + c1 * np.sin(2 * theta_steps)
        )
        # 三维螺线
        x3 = r * np.sin(theta_steps)
        z3 = r * np.cos(theta_steps)
        y3 = y2

        an31.set_data(x3[0 : 90 + 40], y3[0 : 90 + 40])
        an31.set_3d_properties(z3[0 : 90 + 40])
        an32.set_data(x3[90:320], y3[90:320])
        an32.set_3d_properties(z3[90:320])
        an33.set_data(x3[320 : 450 + 40], y3[320 : 450 + 40])
        an33.set_3d_properties(z3[320 : 450 + 40])
        an34.set_data(x3[450:660], y3[450:660])
        an34.set_3d_properties(z3[450:660])
        an35.set_data(x3[660:], y3[660:])
        an35.set_3d_properties(z3[660:])

    def update(c0):
        update_2d(c0)
        update_3d(c0)
        return an1, an31, an32, an33, an34, an35

    ani = animation.FuncAnimation(
        fig, update, np.concatenate((np.linspace(-0.3, 0.3, 25),np.linspace(0.3, -0.3, 25))), interval=10, blit=True
    )


# 二维面板设置
if True:
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.set_xlim(-2.5 * 2 * np.pi, 2.5 * 2 * np.pi)
    ax.set_ylim(-4, 4)

    ax.text(15, -1, "θ", fontsize=15)
    ax.text(1, 4, "z", fontsize=15)

# 绘制三维圆柱 （2020年11月24日 加上视角）
if True:
    # 定义三维数据
    ksi_steps = np.linspace(0, 2 * np.pi, ksi_number)
    # 圆柱数据
    xx = r * np.cos(ksi_steps)
    zz = r * np.sin(ksi_steps)
    yy = np.linspace(0, length, z_number)
    X, Y = np.meshgrid(xx, yy)
    Z, _ = np.meshgrid(zz, yy)

    # 圆柱作图
    ax3.plot_surface(X, Y, Z, color="blue", alpha=0.1)
    ax3.plot_wireframe(X, Y, Z, rstride=100, cstride=125, color="grey", alpha=0.1)

    # 扩大图形范围，让圆柱长一点
    ax3.plot3D(-2, 0, -2)
    ax3.plot3D(2, 0, 2)

    # 坐标
    ax3.plot3D([0, r * 2], [0, 0], [0, 0], color="k", lw=0.8)
    ax3.plot3D([0, 0], [0, 0], [0, r * 2], color="k", lw=0.8)
    ax3.plot3D([0, 0], [0, length], [0, 0], color="k", lw=0.8)

    plt.axis("off")
    ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # 设置摄像机位置（实际上是设置坐标轴范围）
    ax3.set_xlim3d(-1.5, 1.5)
    ax3.set_ylim3d(0.5, 3.5)
    ax3.set_zlim3d(-1.2, 1.2)

    # 设置摄像机方位角
    # ax3.view_init(elev=29, azim=-60)
    ax3.view_init(elev=43, azim=-26)

    # 文字
    ax3.text(0, 3, 0, "z", (0, 1, 0), fontsize=15)
    ax3.text(2 * r, 0, 0, "y", (1, 0, 0), fontsize=15)
    ax3.text(0, 0, r * 2, "x", (0, 0, 0), fontsize=15)


# 保存动画
ani.save("四极CCT核幅值.gif", writer="imagemagick", fps=100)
plt.show()