
from espresso import * # First import the module
from ase.atoms import Atoms, Atom # Import the atoms object from ASE

atoms = Atoms([Atom('H', (0, 0, 0))],
              cell = (8, 9, 10))

with Espresso('output/H',                  # With respect to the directory this script
                                           # is in, this is the directory where the
                                           # calculation will be taking place. The module
                                           # will automatically make the folders necessary.
                                           # Just assure the folder doesn't exist, and if it
                                           # does, that it's empty

              atoms=atoms,                 # This is where we put in the atoms object

              ecutwfc=60.0, ecutrho=600.0, # These are the kinetic energy cutoff parameters
                                           # These values determine heavily the convergence
                                           # of your calculation and therefore the time and
                                           # accuracy of your calculation. You should perform
                                           # convergence tests before performing large amounts
                                           # of studies.

              kpts=(1, 1, 1),              # This is how many kpoints in the x, y, and z
                                           # direction of the unit cell. Similar to ecutwfc
                                           # and ecutrho, the more kpoints the more converged
                                           # and expensive. Testing is recommended.

              occupations='smearing',      # This is to determing the smearing at electrons
                                           # at the fermi level. Typically we do smearing.

              smearing='gauss',            # The type of smearing we want. Typically its gauss
                                           # for insulators and mp (methfessel-paxton) for
                                           # metals.

              degauss=0.01) as calc:       # The width of the smearing. Will dicuss this value
                                           # later.
    try:
        calc.calculate()
        print calc.espressodir, 'Complete'
    except (EspressoSubmitted, EspressoRunning):
        print calc.espressodir, 'Running'
    except (EspressoNotConverged):
        print calc.espressodir, 'Not Converged'
