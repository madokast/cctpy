from cctpy import *

ip = ParticleFactory.create_proton_by_position_and_velocity(
    position=P3(8.61354497625899, -3.709405986818126e-07, 0.0),
    velocity=P3(1.6549917112929747e-07, -174317774.94179922, 0.0)
)

print(ip.get_natural_coordinate_system())
print(ip.speed)


rp = ParticleFactory.create_proton_by_position_and_velocity(
    position=P3(11.633100509643555, 4.317243576049805, 2.0744740962982178),
    velocity=P3(185180528.0, 59419448.0, 106335024.0)
)
print(rp.speed)
print(Protons.get_kinetic_energy_MeV(ip.compute_scalar_momentum()))
print(Protons.get_kinetic_energy_MeV(rp.compute_scalar_momentum()))

p = PhaseSpaceParticle.create_from_running_particle(
    ip,ip.get_natural_coordinate_system(),rp
)
print(p)

# ip=p=(8.61354497625899, -3.709405986818126e-07, 0.0),v=(1.6549917112929747e-07, -174317774.94179922, 0.0),v0=174317774.94179922
# rp=p=(11.633100509643555, 4.317243576049805, 2.0744740962982178),v=(185180528.0, 59419448.0, 106335024.0),v0=177165952.0
# lcs=LOCATION=(8.61354497625899, -3.709405986818126e-07, 0.0), xi=(1.0, 9.494107596574928e-16, -0.0), yi=(0.0, 0.0, 1.0), zi=(9.494107596574928e-16, -1.0, 0.0)

v = P3(185180528.0, 59419448.0, 106335024.0)
print(v.length())