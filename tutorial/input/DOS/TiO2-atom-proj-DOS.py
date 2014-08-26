
from espresso import *
from ase_addons.bulk import rutile
import matplotlib.pyplot as plt

atoms = rutile(('Ti', 'O'), a=4.65, c=2.97, u=0.31)

Ti_ind = [0, 1]
O_ind = [2, 3, 4, 5]

with Espresso('output/TiO2', atoms=atoms, wf_collect=True,
              ecutwfc=40.0, ecutrho=500.0, kpts=(6, 6, 6),
              occupations='smearing', smearing='mp', degauss=0.01,
              nspin=2, walltime='24:00:00', ppn=4) as calc:
    fermi=calc.get_fermi_level()
    dos = EspressoDos(efermi=fermi)
    E = dos.get_energies()
    occupied = (E < 0)
    d, p = np.zeros(len(E)), np.zeros(len(E))
    for i in Ti_ind:
        d += dos.get_site_dos(i, '3d')
    for i in O_ind:
        p += dos.get_site_dos(i, '2p')

plt.plot(E, p, c='r', label='O 2p')
plt.fill_between(x=E[occupied], y1=p[occupied],
                 y2=np.zeros(p[occupied].shape),
                 color='pink')
plt.plot(E, d, c='b', label='Ti 3d')
plt.fill_between(x=E[occupied], y1=d[occupied],
                 y2=np.zeros(d[occupied].shape),
                 color='lightblue')
plt.xlim(-10, 5)
plt.ylim(0, 12)
plt.xlabel('Energy (eV)')
plt.ylabel('DOS')
plt.legend(loc=2)
plt.savefig('figures/TiO2-DOS.png')
plt.show()
