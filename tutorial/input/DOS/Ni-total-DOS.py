
from espresso import *
from ase.lattice import bulk
from ase.visualize import view
import matplotlib.pyplot as plt

atoms = bulk('Ni', 'fcc')

with Espresso('output/Ni', atoms=atoms, wf_collect=True,
              ecutwfc=40.0, ecutrho=500.0, kpts=(6, 6, 6),
              occupations='smearing', smearing='mp', degauss=0.01,
              nspin=2) as calc:
    fermi = calc.get_fermi_level()
    dos = EspressoDos(efermi=fermi) # Initialize the EspressoDos class which contains
                                    # all of the information needed to construct the
                                    # DOS
    E = dos.get_energies()          # Read an array of energies in which the DOS is constructed

    d = dos.get_total_dos()         # Read the density of states at each energy in E

    ind = (E < 5) & (E > -10)       # We're only concerned with the energies
                                    # near the fermi level

    occupied = (E < 0) & (E > -10)  # These are the occupied energy levels

plt.plot(E[ind], d[ind])        # Code for plotting the density of states
plt.fill_between(x=E[occupied], y1=d[occupied],
                 y2=np.zeros(d[occupied].shape), color='lightblue')                    
plt.xlim(-10, 5)
plt.ylim(0, 6)
plt.xlabel('Energy (eV)')
plt.ylabel('DOS (arbitrary units)')
plt.savefig('figures/Ni-total-DOS.png')
plt.show()
