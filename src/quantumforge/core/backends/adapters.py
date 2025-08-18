"""Backend adapter interfaces for quantum chemistry codes."""

import abc
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

    from ..functional_base import FunctionalBase
    from ..grid import GridBase


class BackendAdapterBase(abc.ABC):
    """Abstract base class for quantum chemistry backend adapters."""

    def __init__(
        self,
        device: Optional["torch.device"] = None,
        scratch_dir: Optional[Path] = None,
    ) -> None:
        """Initialize backend adapter."""
        # Import torch at runtime to avoid import errors in test environment
        try:
            import torch
            self.device = device or torch.device("cpu")
        except ImportError:
            self.device = "cpu"  # Fallback for testing

        self.scratch_dir = scratch_dir or Path.cwd() / "scratch"
        self.scratch_dir.mkdir(exist_ok=True)

    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the backend."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and properly configured."""

    @abc.abstractmethod
    def setup_calculation(
        self,
        atoms: List[Tuple[str, Tuple[float, float, float]]],
        basis: str,
        functional: Optional["FunctionalBase"] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set up a quantum chemistry calculation."""

    @abc.abstractmethod
    def run_scf(
        self,
        config: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run self-consistent field calculation."""

    @abc.abstractmethod
    def extract_density(
        self,
        result: Dict[str, Any],
        grid: Optional["GridBase"] = None,
        **kwargs: Any,
    ) -> Any:
        """Extract electron density from calculation results."""

    @abc.abstractmethod
    def extract_density_gradient(
        self,
        result: Dict[str, Any],
        grid: Optional["GridBase"] = None,
        **kwargs: Any,
    ) -> Any:
        """Extract density gradient from calculation results."""


class PySCFAdapter(BackendAdapterBase):
    """Adapter for PySCF quantum chemistry package."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mol = None
        self._mf = None

    def name(self) -> str:
        """Return the name of the backend."""
        return "PySCF"

    def is_available(self) -> bool:
        """Check if PySCF is available."""
        try:
            import pyscf  # noqa: F401
            return True
        except ImportError:
            return False

    def setup_calculation(
        self,
        atoms: List[Tuple[str, Tuple[float, float, float]]],
        basis: str,
        functional: Optional["FunctionalBase"] = None,
        charge: int = 0,
        spin: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set up PySCF calculation."""
        try:
            from pyscf import dft, gto
        except ImportError as exc:
            raise ImportError("PySCF is required for PySCFAdapter") from exc

        # Create molecule
        atom_string = "\n".join([
            f"{symbol} {x} {y} {z}"
            for symbol, (x, y, z) in atoms
        ])

        mol = gto.Mole()
        mol.atom = atom_string
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.build()

        # Create mean-field object
        if functional:
            # Use custom functional
            mf = dft.RKS(mol)
            # Note: Custom functional integration would go here
        else:
            # Use default functional
            from pyscf import scf
            mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)

        self._mol = mol
        self._mf = mf

        return {
            "mol": mol,
            "mf": mf,
            "basis": basis,
            "charge": charge,
            "spin": spin,
        }

    def run_scf(
        self,
        config: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run PySCF SCF calculation."""
        mf = config["mf"]

        # Run SCF
        energy = mf.scf(**kwargs)

        # Convert results to PyTorch tensors
        result = {
            "energy": self._to_tensor(energy),
        }

        # Get density matrix if needed
        if hasattr(mf, "make_rdm1"):
            dm = mf.make_rdm1()
            result["density_matrix"] = self._to_tensor(dm)

        # Store molecular orbitals
        if hasattr(mf, "mo_coeff"):
            result["mo_coeff"] = self._to_tensor(mf.mo_coeff)
            result["mo_energy"] = self._to_tensor(mf.mo_energy)

        return result

    def extract_density(
        self,
        result: Dict[str, Any],
        grid: Optional["GridBase"] = None,
        **kwargs: Any,
    ) -> Any:
        """Extract electron density from PySCF results."""
        if "density_matrix" not in result:
            raise ValueError("Density matrix not found in results")

        if grid is None:
            # Return density matrix directly
            return result["density_matrix"]

        # Evaluate density on grid points
        from pyscf import dft

        coords = grid.get_coordinates().cpu().numpy()
        dm = result["density_matrix"].cpu().numpy()
        ao_values = dft.numint.eval_ao(self._mol, coords)

        # Evaluate density: rho = sum_ij D_ij * phi_i * phi_j
        rho = dft.numint.eval_rho(self._mol, ao_values, dm)

        return self._to_tensor(rho)

    def extract_density_gradient(
        self,
        result: Dict[str, Any],
        grid: Optional["GridBase"] = None,
        **kwargs: Any,
    ) -> Any:
        """Extract density gradient from PySCF results."""
        if grid is None:
            raise ValueError(
                "Grid is required for density gradient extraction"
            )

        from pyscf import dft

        coords = grid.get_coordinates().cpu().numpy()
        dm = result["density_matrix"].cpu().numpy()
        ao_values = dft.numint.eval_ao(self._mol, coords, deriv=1)

        # eval_rho with deriv=1 returns [rho, grad_x, grad_y, grad_z]
        rho_and_grad = dft.numint.eval_rho(
            self._mol, ao_values, dm, xctype="GGA"
        )
        gradients = rho_and_grad[1:4]  # Extract gradient components

        # Convert to PyTorch
        grad_tensor = self._to_tensor(gradients)
        return grad_tensor.T  # Shape: (n_points, 3)

    def _to_tensor(self, array):
        """Convert numpy array to PyTorch tensor."""
        try:
            import torch
            return torch.tensor(
                array, device=self.device, dtype=torch.float64
            )
        except ImportError:
            # Fallback for testing
            return array


class CP2KAdapter(BackendAdapterBase):
    """Adapter for CP2K quantum chemistry package."""

    def name(self) -> str:
        """Return the name of the backend."""
        return "CP2K"

    def is_available(self) -> bool:
        """Check if CP2K is available."""
        try:
            result = subprocess.run(
                ["cp2k.ssmp", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def setup_calculation(
        self,
        atoms: List[Tuple[str, Tuple[float, float, float]]],
        basis: str,
        functional: Optional["FunctionalBase"] = None,
        charge: int = 0,
        spin: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set up CP2K calculation."""
        input_file = self.scratch_dir / "cp2k_input.inp"
        output_file = self.scratch_dir / "cp2k_output.out"

        # Generate CP2K input
        input_content = self._generate_cp2k_input(
            atoms, basis, charge, spin, **kwargs
        )

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(input_content)

        return {
            "input_file": input_file,
            "output_file": output_file,
            "atoms": atoms,
            "basis": basis,
            "charge": charge,
            "spin": spin,
        }

    def run_scf(
        self,
        config: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run CP2K SCF calculation."""
        input_file = config["input_file"]
        output_file = config["output_file"]

        # Run CP2K
        cmd = ["cp2k.ssmp", "-i", str(input_file), "-o", str(output_file)]
        result = subprocess.run(
            cmd,
            cwd=self.scratch_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"CP2K calculation failed: {result.stderr}")

        # Parse energy from output
        energy = self._parse_cp2k_energy(config["output_file"])

        return {
            "energy": self._to_tensor(energy),
            "cp2k_output": config["output_file"],
        }

    def extract_density(
        self,
        result: Dict[str, Any],
        grid: Optional["GridBase"] = None,
        **kwargs: Any,
    ) -> Any:
        """Extract electron density from CP2K results."""
        raise NotImplementedError(
            "CP2K density extraction not yet implemented"
        )

    def extract_density_gradient(
        self,
        result: Dict[str, Any],
        grid: Optional["GridBase"] = None,
        **kwargs: Any,
    ) -> Any:
        """Extract density gradient from CP2K results."""
        raise NotImplementedError(
            "CP2K gradient extraction not yet implemented"
        )

    def _generate_cp2k_input(
        self,
        atoms: List[Tuple[str, Tuple[float, float, float]]],
        basis: str,  # noqa: ARG002
        charge: int,
        spin: int,
        **kwargs: Any,
    ) -> str:
        """Generate CP2K input file content."""
        # Basic CP2K input template
        input_template = f"""
&GLOBAL
  PROJECT_NAME cp2k_calc
  RUN_TYPE ENERGY
&END GLOBAL

&FORCE_EVAL
  METHOD QS
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    CHARGE {charge}
    MULTIPLICITY {spin + 1}
    &MGRID
      CUTOFF 400
    &END MGRID
    &QS
      METHOD GPW
    &END QS
    &SCF
      MAX_SCF 50
      EPS_SCF 1.0E-6
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      ABC 10.0 10.0 10.0
      PERIODIC NONE
    &END CELL
    &COORD
"""

        # Add coordinates
        for symbol, (x, y, z) in atoms:
            input_template += f"      {symbol} {x:.6f} {y:.6f} {z:.6f}\n"

        input_template += """    &END COORD
    &KIND H
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE-q1
    &END KIND
    &KIND C
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE-q4
    &END KIND
    &KIND N
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE-q5
    &END KIND
    &KIND O
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE-q6
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
"""
        return input_template

    def _parse_cp2k_energy(self, output_file: Path) -> float:
        """Parse total energy from CP2K output file."""
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if "ENERGY| Total FORCE_EVAL" in line:
                    # Extract energy value
                    parts = line.split()
                    return float(parts[8])  # Energy is typically 9th column

        raise ValueError("Could not find total energy in CP2K output")

    def _to_tensor(self, value):
        """Convert value to PyTorch tensor."""
        try:
            import torch
            return torch.tensor(
                value, device=self.device, dtype=torch.float64
            )
        except ImportError:
            # Fallback for testing
            return value


def create_backend_adapter(
    backend_name: str,
    **kwargs: Any,
) -> BackendAdapterBase:
    """Factory function to create backend adapters."""
    adapters = {
        "pyscf": PySCFAdapter,
        "cp2k": CP2KAdapter,
    }

    backend_name = backend_name.lower()
    if backend_name not in adapters:
        available = ", ".join(adapters.keys())
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available: {available}"
        )

    adapter_class = adapters[backend_name]
    return adapter_class(**kwargs)


__all__ = [
    "BackendAdapterBase",
    "PySCFAdapter",
    "CP2KAdapter",
    "create_backend_adapter",
]
]
