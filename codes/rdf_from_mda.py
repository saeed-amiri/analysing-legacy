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
                 residue: str,  # To get its rdf
                 box_range: tuple[float, float]  # Where of the box to look
                 ) -> None:
        self.residue = residue
        self.box_range = box_range
        self.uinverse = mda.Universe(sys.argv[1], sys.argv[2])
        self.initiate()

    def initiate(self) -> None:
        """
        get bins and rdf!
        """
        nanoparticle: "AtomGroup" = self.get_nanoparticle()
        target_atoms: "AtomGroup" = self.get_atoms()
        bins: np.ndarray
        rdf:  np.ndarray
        bins, rdf = \
            self.get_rdf(reference=nanoparticle, target=target_atoms)
        self.plot_rdf(bins, rdf)

    def get_atoms(self) -> "AtomGroup":
        """
        find target atoms
        """
        args: str = f"resname {self.residue} and "
        args += f"prop z > {self.box_range[0]} and "
        args += f"prop z < {self.box_range[1]}"
        return self.uinverse.select_atoms(args)

    def get_nanoparticle(self) -> "AtomGroup":
        """
        Only APT is calculated as the NP
        """
        return self.uinverse.select_atoms('resname APT')

    @staticmethod
    def get_rdf(reference: "AtomGroup",
                target: "AtomGroup"
                ) -> tuple[np.ndarray, np.ndarray]:
        """
        return bins and rdf
        """
        rdf = MDAnalysis.analysis.rdf.InterRDF(reference.atoms, target.atoms)
        rdf.run()
        return rdf.results.bins, rdf.results.rdf

    @staticmethod
    def plot_rdf(bins: np.ndarray,
                 rdf: np.ndarray
                 ) -> None:
        """
        plot!
        """
        plt.plot(bins, rdf)
        plt.show()


if __name__ == "__main__":
    GetTraj("ODN", (106, 150))
