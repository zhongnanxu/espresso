
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

    d_u = dos.get_total_dos(spin='+') # Read the spin up density of states at each energy in E
    d_d = dos.get_total_dos(spin='-') # Read the spin down density of states at each energy in E

    ind = (E < 5) & (E > -10)       # We're only concerned with the energies
                                    # near the fermi level

    occupied = (E < 0) & (E > -10)  # These are the occupied energy levels

plt.plot(E[ind], d_u[ind], c='b')        # Code for plotting the density of states
plt.plot(E[ind], -d_d[ind], c='b')       # Code for plotting the density of states
plt.fill_between(x=E[occupied], y1=d_u[occupied],
                 y2=np.zeros(E[occupied].shape), color='lightblue')                    
plt.fill_between(x=E[occupied], y1=-d_d[occupied],
                 y2=np.zeros(E[occupied].shape), color='lightblue')                    

plt.xlim(-10, 5)
plt.ylim(-3, 3)
plt.xlabel('Energy (eV)')
plt.ylabel('DOS (arbitrary units)')
plt.savefig('figures/Ni-total-spin-DOS.png')
plt.show()
