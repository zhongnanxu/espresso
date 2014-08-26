
from espresso import *
from ase.lattice import bulk
import matplotlib.pyplot as plt

atoms = bulk('Ni', 'fcc', 3.52)

orbitals = ['3d', '4s']

colors = {'4s':['orange','y'],    # A dictionary of the orbitals and
          '3d':['b','lightblue']} # colors we want to graph them as.

with Espresso('output/Ni', atoms=atoms, wf_collect=True,
              ecutwfc=40.0, ecutrho=500.0,
              occupations='smearing', smearing='mp', degauss=0.01,
              nspin=2, kpts=(6, 6, 6), walltime='24:00:00', ppn=4) as calc:
    fermi = calc.get_fermi_level()
    dos = EspressoDos(efermi=fermi)
    energies = dos.get_energies()
    occupied = (energies < 0) & (energies > -10)
    for orb in orbitals:
        ind = (energies < 5) & (energies > -10)
        d = dos.get_site_dos(0, orb)
        
        plt.plot(energies[ind], d[ind], c=colors[orb][0], label=orb)
        plt.fill_between(x=energies[occupied], y1=d[occupied], 
                         y2=np.zeros(energies[occupied].shape),
                         color=colors[orb][1], label=orb)
plt.xlabel('Energy (eV)')
plt.ylabel('DOS (arbitrary units)')
plt.ylim(0, 6)
plt.savefig('figures/Ni-spin-proj-DOS.png')
plt.legend()
plt.show()
