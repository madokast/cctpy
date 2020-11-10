import unittest

import numpy as np

from cctpy.baseutils import Vectors, Equal, Stream
from cctpy.constant import M, MM, YI, XI, Protons, ZI, MRAD
from cctpy.particle import RunningParticle, ParticleFactory, PhaseSpaceParticle, ParticleRunner
from cctpy.qs_hard_edge_magnet import QsHardEdgeMagnet
from cctpy.abstract_classes import LocalCoordinateSystem
from cctpy.plotuils import Plot2


class QsTest(unittest.TestCase):
    def test_quad_0(self):
        """
        测试 qs 四极场
        Returns
        -------

        """
        length = 0.2 * M
        aper = 30 * MM
        g = 10.
        L = 0
        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)
        qs = QsHardEdgeMagnet(length, g, L, aper, lc)

        m = qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, 0))
        self.assertTrue(Equal.equal_vector(m, Vectors.create(0.0, 0.0, -0.1)))

        m = qs.magnetic_field_at(Vectors.create(15 * MM, 0.1, 0))
        self.assertTrue(Equal.equal_vector(m, Vectors.create(0.0, 0.0, -0.15)))

        m = qs.magnetic_field_at(Vectors.create(15 * MM, 0.1, 5 * MM))
        self.assertTrue(Equal.equal_vector(m, Vectors.create(-0.05, -3.061616997868383E-18, -0.15)))

    def test_quad_1(self):
        """
        测试 qs 四极场
        Returns
        -------

        """
        length = 0.2 * M
        aper = 30 * MM
        g = -45.7
        L = 0
        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)
        qs = QsHardEdgeMagnet(length, g, L, aper, lc)

        m = qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, 0))
        self.assertTrue(Equal.equal_vector(m, Vectors.create(0.0, 0.0, 0.457)))

        m = qs.magnetic_field_at(Vectors.create(15 * MM, 0.1, 0))
        self.assertTrue(Equal.equal_vector(m, Vectors.create(0.0, 0.0, 0.6855)))

        m = qs.magnetic_field_at(Vectors.create(15 * MM, 0.1, 5 * MM))
        self.assertTrue(Equal.equal_vector(m, Vectors.create(0.2285, 1.399158968025851E-17, 0.6855)))

    def test_second_0(self):
        length = 0.2 * M
        aper = 30 * MM
        g = 0
        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)

        mx = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, 0))).map(lambda m: m[0]).to_vector()

        my = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, 0))).map(lambda m: m[1]).to_vector()

        mz = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, 0))).map(lambda m: m[2]).to_vector()

        self.assertTrue(Equal.equal_vector(mx, np.array([-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
        self.assertTrue(Equal.equal_vector(my, np.array([-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
        self.assertTrue(Equal.equal_vector(mz, np.array(
            [-0.005, -0.0038888888888888888, -0.002777777777777778, -0.0016666666666666672, -5.555555555555558E-4,
             5.555555555555558E-4, 0.001666666666666666, 0.0027777777777777775, 0.0038888888888888888, 0.005])))

    def test_second_1(self):
        length = 0.2 * M
        aper = 30 * MM
        g = 0
        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)

        mx = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, 1 * MM))).map(lambda m: m[0]).to_vector()

        my = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, 1 * MM))).map(lambda m: m[1]).to_vector()

        mz = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, 1 * MM))).map(lambda m: m[2]).to_vector()

        self.assertTrue(Equal.equal_vector(mx, np.array(
            [-0.001, -7.777777777777777E-4, -5.555555555555557E-4, -3.3333333333333343E-4, -1.1111111111111116E-4,
             1.1111111111111116E-4, 3.3333333333333316E-4, 5.555555555555554E-4, 7.777777777777777E-4, 0.001]
        )))
        self.assertTrue(Equal.equal_vector(my, np.array(
            [-6.123233995736766E-20, -4.762515330017485E-20, -3.4017966642982043E-20, -2.0410779985789227E-20,
             -6.80359332859641E-21, 6.80359332859641E-21, 2.041077998578921E-20, 3.4017966642982025E-20,
             4.762515330017485E-20, 6.123233995736766E-20]
        )))
        self.assertTrue(Equal.equal_vector(mz, np.array(
            [-0.00495, -0.00385, -0.0027500000000000003, -0.0016500000000000006, -5.500000000000002E-4,
             5.500000000000002E-4, 0.0016499999999999991, 0.0027499999999999994, 0.00385, 0.00495])))

    def test_second_2(self):
        length = 0.2 * M
        aper = 30 * MM
        g = 0
        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)

        mx = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, -1 * MM))).map(lambda m: m[0]).to_vector()

        my = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, -1 * MM))).map(lambda m: m[1]).to_vector()

        mz = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(10 * MM, 0.1, -1 * MM))).map(lambda m: m[2]).to_vector()

        self.assertTrue(Equal.equal_vector(mx, np.array(
            [0.001, 7.777777777777777E-4, 5.555555555555557E-4, 3.3333333333333343E-4, 1.1111111111111116E-4,
             -1.1111111111111116E-4, -3.3333333333333316E-4, -5.555555555555554E-4, -7.777777777777777E-4, -0.001]
        )))
        self.assertTrue(Equal.equal_vector(my, np.array(
            [6.123233995736766E-20, 4.762515330017485E-20, 3.4017966642982043E-20, 2.0410779985789227E-20,
             6.80359332859641E-21, -6.80359332859641E-21, -2.041077998578921E-20, -3.4017966642982025E-20,
             -4.762515330017485E-20, -6.123233995736766E-20]
        )))
        self.assertTrue(Equal.equal_vector(mz, np.array(
            [-0.00495, -0.00385, -0.0027500000000000003, -0.0016500000000000006, -5.500000000000002E-4,
             5.500000000000002E-4, 0.0016499999999999991, 0.0027499999999999994, 0.00385, 0.00495]
        )))

    def test_second_3(self):
        length = 0.2 * M
        aper = 30 * MM
        g = 0
        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)

        mx = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(-5 * MM, 0.1, -1 * MM))).map(lambda m: m[0]).to_vector()

        my = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(-5 * MM, 0.1, -1 * MM))).map(lambda m: m[1]).to_vector()

        mz = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(-5 * MM, 0.1, -1 * MM))).map(lambda m: m[2]).to_vector()

        self.assertTrue(Equal.equal_vector(mx, np.array(
            [-5.0E-4, -3.8888888888888887E-4, -2.7777777777777783E-4, -1.6666666666666672E-4, -5.555555555555558E-5,
             5.555555555555558E-5, 1.6666666666666658E-4, 2.777777777777777E-4, 3.8888888888888887E-4, 5.0E-4]
        )))
        self.assertTrue(Equal.equal_vector(my, np.array(
            [-3.061616997868383E-20, -2.3812576650087424E-20, -1.7008983321491022E-20, -1.0205389992894614E-20,
             -3.401796664298205E-21, 3.401796664298205E-21, 1.0205389992894605E-20, 1.7008983321491013E-20,
             2.3812576650087424E-20, 3.061616997868383E-20]
        )))
        self.assertTrue(Equal.equal_vector(mz, np.array(
            [-0.0012000000000000001, -9.333333333333333E-4, -6.666666666666668E-4, -4.0000000000000013E-4,
             -1.3333333333333337E-4, 1.3333333333333337E-4, 3.999999999999998E-4, 6.666666666666665E-4,
             9.333333333333333E-4, 0.0012000000000000001]
        )))

    def test_second_4(self):
        length = 0.2 * M
        aper = 30 * MM
        g = 0
        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)

        mx = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(-8 * MM, 0.1, 1 * MM))).map(lambda m: m[0]).to_vector()

        my = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(-8 * MM, 0.1, 1 * MM))).map(lambda m: m[1]).to_vector()

        mz = Stream.linspace(-100, 100, 10).map(lambda k: QsHardEdgeMagnet(length, g, k, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(Vectors.create(-8 * MM, 0.1, 1 * MM))).map(lambda m: m[2]).to_vector()

        self.assertTrue(Equal.equal_vector(mx, np.array(
            [7.999999999999999E-4, 6.222222222222221E-4, 4.444444444444444E-4, 2.6666666666666673E-4,
             8.88888888888889E-5, -8.88888888888889E-5, -2.666666666666665E-4, -4.444444444444443E-4,
             -6.222222222222221E-4, -7.999999999999999E-4]
        )))
        self.assertTrue(Equal.equal_vector(my, np.array(
            [4.8985871965894125E-20, 3.8100122640139875E-20, 2.7214373314385626E-20, 1.632862398863138E-20,
             5.442874662877126E-21, -5.442874662877126E-21, -1.6328623988631368E-20, -2.721437331438562E-20,
             -3.8100122640139875E-20, -4.8985871965894125E-20]
        )))
        self.assertTrue(Equal.equal_vector(mz, np.array(
            [-0.00315, -0.00245, -0.00175, -0.0010500000000000004, -3.500000000000001E-4, 3.500000000000001E-4,
             0.0010499999999999995, 0.0017499999999999996, 0.00245, 0.00315]
        )))

    def test_quad_and_second_0(self):
        length = 0.2 * M
        aper = 30 * MM
        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)

        p = Vectors.create(-8 * MM, 0.1, 1 * MM)

        mx = Stream.linspace(-100, 100, 10).map(
            lambda k: QsHardEdgeMagnet(length, np.sin(k / 180) * 20, (1.1 ** (k / 2)) * 2, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(p)).map(lambda m: m[0]).to_vector()

        my = Stream.linspace(-100, 100, 10).map(
            lambda k: QsHardEdgeMagnet(length, np.sin(k / 180) * 20, (1.1 ** (k / 2)) * 2, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(p)).map(lambda m: m[1]).to_vector()

        mz = Stream.linspace(-100, 100, 10).map(
            lambda k: QsHardEdgeMagnet(length, np.sin(k / 180) * 20, (1.1 ** (k / 2)) * 2, aper, lc)).map(
            lambda qs: qs.magnetic_field_at(p)).map(lambda m: m[2]).to_vector()

        self.assertTrue(Equal.equal_vector(mx, np.array(
            [0.01054817141861684, 0.008375158765307863, 0.006074168017454833, 0.0036793034142077055,
             0.0012243616386317005, -0.0012609533747681091, -0.003760913727964488, -0.006301201552951284,
             -0.009026933421046367, -0.012426561361512444]
        )))
        self.assertTrue(Equal.equal_vector(my, np.array(
            [6.458892182333354E-19, 5.128305687142586E-19, 3.7193552100296426E-19, 2.2529235746506973E-19,
             7.497052808745602E-20, -7.721112571419089E-20, -2.302895479410525E-19, -3.8583731563020608E-19,
             -5.52740256010035E-19, -7.609074297892195E-19]
        )))
        self.assertTrue(Equal.equal_vector(mz, np.array(
            [-0.08438592505476789, -0.06700286672870447, -0.04859794794067867, -0.029447702336309195,
             -0.009833171528290597, 0.009977251489327706, 0.029769042946726516, 0.04949189248669594,
             0.06956922943567483, 0.09178208545491932]
        )))

    def test_track_y(self):
        """
        六级 qs track 对比 y 方向
        Returns
        -------

        """

        plane = PhaseSpaceParticle.YYP_PLANE
        delta = 0.
        number = 6

        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)
        qs = QsHardEdgeMagnet(0.2, 0, 10000 * 2, 300000 * MM, lc)

        rp = ParticleFactory.create_proton(
            Vectors.create(0, -0.5, 0), YI
        )

        # print(f"rp={rp}")

        pps = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
            plane, 3.5 * MM, 7.2 * MM, delta, number
        )

        # print(*pps, sep='\n', end='\n\n')

        pp = ParticleFactory.create_from_phase_space_particles(
            rp, rp.get_natural_coordinate_system(y_direction=ZI), pps
        )

        # print(*pp, sep='\n\n')

        ParticleRunner.run_ps_only_cpu0(pp, qs, 1.2)
        ParticleRunner.run_only(rp, qs, 1.2)

        # print(f"rp={rp}")
        # print(*pp, sep='\n\n')

        pps_end = PhaseSpaceParticle.create_from_running_particles(rp, rp.get_natural_coordinate_system(), pp)

        li = PhaseSpaceParticle.phase_space_particles_project_to_plane(pps_end, plane)

        li = np.array(
            [[x / MM, xp / MRAD] for x, xp in li]
        )

        x = li[:, 0]
        y = li[:, 1]

        x0 = np.array(
            [4.571009592873671, 13.005311328487931, 4.473631539146663, -4.5763158484424205, -13.005311328486815,
             -4.473631539149022]
            )
        y0 = np.array(
            [1.9535672206449075, 13.092945863265955, 5.607514554681223, -1.9596240807758292, -13.092945863264303,
             -5.6075145546827025]
            )

        self.assertTrue(
            (np.abs(x.flatten() - x0.flatten()) < 0.05).all()
        )

        self.assertTrue(
            (np.abs(y.flatten() - y0.flatten()) < 0.05).all()
        )

        # Plot2.plot2d([(li, 'r.')])
        #
        # Plot2.plot2d([(np.column_stack((x0, y0)), 'k.')])
        #
        # Plot2.show()

    def test_track_x(self):
        """
        六级 QS track 对比 x 方向
        Returns
        -------

        """
        plane = PhaseSpaceParticle.XXP_PLANE
        delta = 0.
        number = 6

        lc = LocalCoordinateSystem(main_direction=YI, second_direction=-XI)
        qs = QsHardEdgeMagnet(0.2, 0, 10000 * 2, 300000 * MM, lc)

        rp = ParticleFactory.create_proton(
            Vectors.create(0, -0.5, 0), YI
        )

        # print(f"rp={rp}")

        pps = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
            plane, 3.5 * MM, 7.2 * MM, delta, number
        )

        # print(*pps, sep='\n', end='\n\n')

        pp = ParticleFactory.create_from_phase_space_particles(
            rp, rp.get_natural_coordinate_system(y_direction=ZI), pps
        )

        # print(*pp, sep='\n\n')

        ParticleRunner.run_ps_only_cpu0(pp, qs, 1.2)
        ParticleRunner.run_only(rp, qs, 1.2)

        # print(f"rp={rp}")
        # print(*pp, sep='\n\n')

        pps_end = PhaseSpaceParticle.create_from_running_particles(rp, rp.get_natural_coordinate_system(), pp)

        li = PhaseSpaceParticle.phase_space_particles_project_to_plane(pps_end, plane)

        li = np.array(
            [[x / MM, xp / MRAD] for x, xp in li]
        )

        x = li[:, 0]
        y = li[:, 1]

        x0 = np.array(
            [-1.6363082716640025, -2.964662344582841, 3.848704911140664, -10.799631136000919, -29.05411958099093,
             -5.103782688758285]
        )
        y0 = np.array(
            [-8.439377477318738, -14.025193841237206, 4.508357473356099, -12.381715740031598, -40.36236303026269,
             -6.717141236931342]
        )

        self.assertTrue(
            (np.abs(x.flatten() - x0.flatten()) < 0.05).all()
        )

        self.assertTrue(
            (np.abs(y.flatten() - y0.flatten()) < 0.05).all()
        )

        # Plot2.plot2d([(li, 'r.')])
        #
        # Plot2.plot2d([(np.column_stack((x0, y0)), 'k.')])
        #
        # Plot2.show()


if __name__ == '__main__':
    unittest.main()
