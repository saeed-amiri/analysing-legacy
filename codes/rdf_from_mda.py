"""
Calculate and RDF by using MDAnalysis
"""
import sys
import typing
import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
import MDAnalysis.analysis.rdf

if typing.TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup


class GetTraj:
    """
    read trajectory by MDAnalysis
    """

    def __init__(self,
                 residues: list[str],  # To get their rdf
                 box_range: tuple[float, float]  # Where of the box to look
                 ) -> None:
        self.residues = residues
        self.box_range = box_range
        self.uinverse = mda.Universe(sys.argv[1], sys.argv[2])
        self.bins_rdfs: list[tuple[np.ndarray, np.ndarray]] = \
            self.initiate()

    def initiate(self) -> None:
        """
        get bins and rdf!
        """
        nanoparticle: "AtomGroup" = self.get_nanoparticle()
        target_atoms: list["AtomGroup"] = self.get_atoms()
        bins_rdfs: list[tuple[np.ndarray, np.ndarray]]
        bins_rdfs = \
            self.get_rdf(reference=nanoparticle, targets=target_atoms)
        return bins_rdfs

    def get_atoms(self) -> list["AtomGroup"]:
        """
        find target atoms
        """
        atoms_groups: list["AtomGroup"] = []
        for residue in self.residues:
            args: str = f"resname {residue} and "
            args += f"prop z > {self.box_range[0]} and "
            args += f"prop z < {self.box_range[1]}"
            atoms_groups.append(self.uinverse.select_atoms(args))
        return atoms_groups

    def get_nanoparticle(self) -> "AtomGroup":
        """
        Only APT is calculated as the NP
        """
        return self.uinverse.select_atoms('resname APT')

    @staticmethod
    def get_rdf(reference: "AtomGroup",
                targets: list["AtomGroup"]
                ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        return bins and rdf
        """
        bins_rdfs: list[tuple[np.ndarray, np.ndarray]] = []
        for target in targets:
            rdf = \
                MDAnalysis.analysis.rdf.InterRDF(reference.atoms, target.atoms)
            rdf.run()
            bins_rdfs.append((rdf.results.bins, rdf.results.rdf))
        return bins_rdfs


if __name__ == "__main__":
    GetTraj(["ODN", "CLA"], (106, 150))
