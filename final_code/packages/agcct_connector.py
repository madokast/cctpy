import numpy
from packages.cct import *


class AGCCT_CONNECTOR(Magnet):
    def __init__(self, agcct1: CCT, agcct2: CCT, step: float = 1*MM) -> None:
        # fields
        # self.current
        # self.local_coordinate_system
        # self.length
        # self.dispersed_path3

        # 必要的验证
        if agcct1.local_coordinate_system != agcct2.local_coordinate_system:
            raise ValueError(
                f"需要连接的cct不属于同一坐标系，无法连接。agccct1={agcct1},agcct2={agcct2}")
        BaseUtils.equal(agcct1.big_r, agcct2.big_r,
                        msg=f"需要连接的 cct big_r 不同，无法连接。agccct1={agcct1},agcct2={agcct2}")
        BaseUtils.equal(agcct1.small_r, agcct2.small_r,
                        msg=f"需要连接的 cct small_r 不同，无法连接。agccct1={agcct1},agcct2={agcct2}")
        BaseUtils.equal(agcct1.current, agcct2.current,
                        msg=f"需要连接的 cct 电流不同，无法连接。agccct1={agcct1},agcct2={agcct2}")
        self.current = agcct1.current
        # 坐标系
        self.local_coordinate_system = agcct1.local_coordinate_system

        # 前 cct 的终点
        pre_cct_end_point_in_ksi_phi_coordinate = agcct1.end_point_in_ksi_phi_coordinate
        # 后 cct 的起点
        next_cct_starting_point_in_ksi_phi_coordinate = agcct2.starting_point_in_ksi_phi_coordinate
        BaseUtils.equal(pre_cct_end_point_in_ksi_phi_coordinate.x, next_cct_starting_point_in_ksi_phi_coordinate.x,
                        msg=f"需要连接的前段 cct 终点 ξ 不等于后段 cct 的起点 ξ，无法连接。agccct1={agcct1},agcct2={agcct2}")

        # 前 cct 的终点，在 φR-ξr 坐标系中的点 a
        # 后 cct 的起点  b
        # 以及切向 va vb
        a = P2(x=pre_cct_end_point_in_ksi_phi_coordinate.y*agcct1.big_r, y=0.0)
        b = P2(x=next_cct_starting_point_in_ksi_phi_coordinate.y*agcct2.big_r, y=0.0)
        va = BaseUtils.derivative(lambda ksi: P2(
            x=agcct1.p2_function(ksi).y*agcct1.big_r,
            y=agcct1.p2_function(ksi).x*agcct1.small_r))(
            pre_cct_end_point_in_ksi_phi_coordinate.x
            # 下行的乘法很重要，因为自变量不一定是随着绕线增大，如果反向绕线就是减小 fixed 2021年1月8日
        ) *(-1 if agcct1.starting_point_in_ksi_phi_coordinate.x > agcct1.end_point_in_ksi_phi_coordinate.x else 1)
        vb = BaseUtils.derivative(lambda ksi: P2(
            x=agcct2.p2_function(ksi).y*agcct2.big_r,
            y=agcct2.p2_function(ksi).x*agcct2.small_r))(
            next_cct_starting_point_in_ksi_phi_coordinate.x
        )*(-1 if agcct2.starting_point_in_ksi_phi_coordinate.x > agcct2.end_point_in_ksi_phi_coordinate.x else 1)
        print(f"a={a}, b={b}")
        print(f"va={va}, vb={vb}")

        # 开始连接
        # 首先是 z-Ξ 坐标系
        ip, ka, kb = StraightLine2.intersecting_point(
            pa=a,
            va=va.copy().rotate(BaseUtils.angle_to_radian(90)),
            pb=b,
            vb=vb
        )
        if kb > 0:
            ip, ka, kb = StraightLine2.intersecting_point(
                pa=a,
                va=va,
                pb=b,
                vb=vb.copy().rotate(BaseUtils.angle_to_radian(90))
            )
            assert ka >= 0, 'ka 应该非负'
            ca = a + ka*va

            connector_line_in_z_Θ = (
                Trajectory
                .set_start_point(start_point=a)
                .first_line(
                    direct=va, length=abs(ka*va.length()))
                .add_arc_line(
                    radius=(ca-b).length()/2,
                    clockwise=StraightLine2.is_on_right(ca, va, b) == 1,
                    angle_deg=180.0
                )
            )
        else:  # kb<=0:
            cb = b + kb*vb
            connector_line_in_z_Θ = (
                Trajectory
                .set_start_point(start_point=a)
                .first_line(direct=va, length=0.0).add_arc_line(
                    radius=(a-cb).length()/2,
                    clockwise=StraightLine2.is_on_right(a, va, b) == 1,
                    angle_deg=180.0
                ).add_strait_line(length=abs(kb*vb.length()))
            )

        self.connector_line_in_z_Θ = connector_line_in_z_Θ
        self.length_in_z_Θ = connector_line_in_z_Θ.get_length()
        self.p2_function = lambda t: P2(
            x=self.connector_line_in_z_Θ.point_at(t).y/agcct1.small_r,
            y=self.connector_line_in_z_Θ.point_at(t).x/agcct1.big_r
        )
        self.p2_function_start = 0
        self.p2_function_end = self.length_in_z_Θ

        # z-Ξ 转到 ksi - phi 坐标系
        # z = φR
        # Ξ = ξr
        # 再转到三维坐标系 xyz
        dispersed_path3 = [agcct1.bipolar_toroidal_coordinate_system.convert(P2(
            x=p2.y/agcct1.small_r,  # p2.y 为 Ξ，ξ = Ξ/r
            y=p2.x/agcct1.big_r  # p2.x 为 z，φ = z/R
        )).to_list() for p2 in self.connector_line_in_z_Θ.disperse2d(step=step)]

        # 转为 numpy 数组
        self.dispersed_path3: numpy.ndarray = numpy.array(dispersed_path3)

        self.elementary_current = 1e-7 * self.current * (
            self.dispersed_path3[1:] - self.dispersed_path3[:-1]
        )

        # 电流元的位置 (p[i+1]+p[i])/2
        self.elementary_current_position = 0.5 * (
            self.dispersed_path3[1:] + self.dispersed_path3[:-1]
        )

    def magnetic_field_at(self, point: P3) -> P3:
        if BaseUtils.equal(self.current, 0, err=1e-6):
            return P3.zeros()

        # point 转为局部坐标，并变成 numpy 向量
        p = numpy.array(
            self.local_coordinate_system.point_to_local_coordinate(
                point).to_list()
        )

        # 点 p 到电流元中点
        r = p - self.elementary_current_position

        # 点 p 到电流元中点的距离的三次方
        rr = (numpy.linalg.norm(r, ord=2, axis=1)
              ** (-3)).reshape((r.shape[0], 1))

        # 计算每个电流元在 p 点产生的磁场 (此时还没有乘系数 μ0/4π )
        dB = numpy.cross(self.elementary_current, r) * rr

        # 求和，即得到磁场，
        # (不用乘乘以系数 μ0/4π = 1e-7)
        # refactor v0.1.1
        B = numpy.sum(dB, axis=0)

        # 转回 P3
        B_P3: P3 = P3.from_numpy_ndarry(B)

        # 从局部坐标转回全局坐标
        B_P3: P3 = self.local_coordinate_system.vector_to_global_coordinate(
            B_P3)

        return B_P3
