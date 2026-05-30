import numpy as np
from ase.io import write


def feco_randamize_generater(
    slab_feco_base,
    n_co,
    supercell=(5, 5, 1),
    seed=0,
    write_struct=False,
    output_path="slab_feco_random.xsf",
):
    rng = np.random.default_rng(seed)

    slab_feco = slab_feco_base.repeat(supercell)

    # z座標で層を判定
    z = np.round(slab_feco.get_positions()[:, 2], 5)
    unique_z = np.unique(z)
    unique_z.sort()

    # 1MLだけ使うため、一番上の層だけ残す
    top_z = unique_z[-1]
    top_indices = np.where(z == top_z)[0]
    slab_feco_1ml = slab_feco[top_indices]

    n_sites = len(slab_feco_1ml)

    if n_co < 0:
        raise ValueError("n_co must be >= 0")

    if n_co > n_sites:
        raise ValueError(
            f"n_co={n_co} is too large. "
            f"1ML has only {n_sites} atomic sites."
        )

    # まず全てFeにする
    symbols = ["Fe"] * n_sites

    # 指定した数だけCoにする
    co_indices = rng.choice(
        np.arange(n_sites),
        size=n_co,
        replace=False
    )

    for i in co_indices:
        symbols[i] = "Co"

    slab_feco_1ml.set_chemical_symbols(symbols)

    if write_struct:
        write(output_path, slab_feco_1ml)

    print(
        f"[feco_randamize_generater] 1ML sites: {n_sites}, "
        f"Co: {n_co}, Fe: {n_sites - n_co}"
    )

    return slab_feco_1ml