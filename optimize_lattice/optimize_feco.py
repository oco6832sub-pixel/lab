import numpy as np
from ase import Atoms
from ase.io import write
from ase.eos import EquationOfState
from gpaw import GPAW

# ================= parameters =================
kpts = (8, 8, 8)
seed = 0

a_list = np.linspace(2.75, 3.00, 9)

result_file = "FeCo_lattice_results.txt"

volumes = []
energies = []
lattice_constants = []

# ================= prepare result file =================
with open(result_file, "w") as f:
    f.write("FeCo lattice constant optimization results\n")
    f.write("Structure: B2 FeCo\n")
    f.write("----------------------------------------\n")
    f.write("a[A]        volume[A^3]        energy[eV]\n")

# ================= EOS calculation =================
for a in a_list:
    atoms = Atoms(
        symbols=["Fe", "Co"],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5)
        ],
        cell=[a, a, a],
        pbc=True
    )

    atoms.set_initial_magnetic_moments([2.2, 1.7])

    atoms.calc = GPAW(
        mode={"name": "lcao"},
        basis="dzp",
        xc="PBE",

        mixer={
            "backend": "pulay",
            "beta": 0.02,
            "nmaxold": 3,
            "weight": 10
        },

        # convergence={
        #     "energy": 1e-3,
        #     "density": 1e-3,
        #     "eigenstates": 1e-3
        # },

        txt=f"output_a_{a:.3f}_seed_{seed}.txt",
        kpts=kpts,
        symmetry="off",
        nbands="nao",

        occupations={
            "name": "fermi-dirac",
            "width": 0.2
        },

        maxiter=300,
        hund=True,
        spinpol=True
    )

    energy = atoms.get_potential_energy()
    volume = atoms.get_volume()

    lattice_constants.append(a)
    energies.append(energy)
    volumes.append(volume)

    print(f"a = {a:.4f} A, volume = {volume:.4f} A^3, energy = {energy:.8f} eV")

    with open(result_file, "a") as f:
        f.write(f"{a:.6f}    {volume:.6f}    {energy:.10f}\n")

# ================= find lowest calculated energy =================
energies_np = np.array(energies)
lattice_np = np.array(lattice_constants)
volumes_np = np.array(volumes)

min_index = np.argmin(energies_np)

a_min_calc = lattice_np[min_index]
e_min_calc = energies_np[min_index]
v_min_calc = volumes_np[min_index]

# ================= EOS fitting =================
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()

a_opt_eos = v0 ** (1 / 3)

# ================= print results =================
print("\n===== Lowest calculated point =====")
print(f"Lowest calculated lattice constant a = {a_min_calc:.6f} A")
print(f"Lowest calculated energy = {e_min_calc:.10f} eV")

print("\n===== EOS fitted result =====")
print(f"EOS optimized lattice constant a = {a_opt_eos:.6f} A")
print(f"EOS minimum energy E0 = {e0:.10f} eV")
print(f"Bulk modulus B = {B:.6f} eV/A^3")

# ================= write final results to file =================
with open(result_file, "a") as f:
    f.write("\n----------------------------------------\n")
    f.write("Lowest calculated point\n")
    f.write(f"lowest_calculated_a[A] = {a_min_calc:.6f}\n")
    f.write(f"lowest_calculated_volume[A^3] = {v_min_calc:.6f}\n")
    f.write(f"lowest_calculated_energy[eV] = {e_min_calc:.10f}\n")

    f.write("\nEOS fitted result\n")
    f.write(f"eos_optimized_a[A] = {a_opt_eos:.6f}\n")
    f.write(f"eos_minimum_energy[eV] = {e0:.10f}\n")
    f.write(f"bulk_modulus[eV/A^3] = {B:.6f}\n")

# ================= save optimized structures =================
atoms_min_calc = Atoms(
    symbols=["Fe", "Co"],
    scaled_positions=[
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5)
    ],
    cell=[a_min_calc, a_min_calc, a_min_calc],
    pbc=True
)
atoms_min_calc.set_initial_magnetic_moments([2.2, 1.7])
write("FeCo_lowest_calculated.xsf", atoms_min_calc)

atoms_eos = Atoms(
    symbols=["Fe", "Co"],
    scaled_positions=[
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5)
    ],
    cell=[a_opt_eos, a_opt_eos, a_opt_eos],
    pbc=True
)
atoms_eos.set_initial_magnetic_moments([2.2, 1.7])
write("FeCo_EOS_optimized.xsf", atoms_eos)

eos.plot("FeCo_EOS.png")