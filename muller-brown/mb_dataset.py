from typing import Optional
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path
import os
from copy import deepcopy
from ase import Atoms, units
from mb_calculator import MullerBrownPotential
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


import torch
from torch.utils.data import Dataset


class MBDataset(Dataset):
    """
    Muller Brown dataset for transition path optimization.
    seed: random seed
    temperature: temperature of the system (Kelvin)
    n_sims: number of simulations to run
    n_steps: number of steps to take in each simulation
    timestep: timestep of the Langevin integrator (fs)
    mass: mass of the atoms # TODO units
    gamma: friction for langevin dynamics (fs^-1)
    save_every: save every nth step (so timestep of dataset is time_step * save_every)
    load_every: load every nth step from the trajectory
    load_forces: whether to load forces from the trajectory or not (adds time in loading)
    default_atom: atom type to use for the atoms
    device: device to run the simulations on
    preload_sim_dir: directory to load simulations
    save_path: path to save generated simulations
    use_langevin: whether to use langevin dynamics or not
    initial_positions: initial positions of the atoms (optional, if provided, will sample uniformly over these)
    calculator: ASE calculator to use for the simulations
    """

    def __init__(
        self,
        seed: int = 0,
        temperature: float = 450.0,
        n_sims: int = 100,
        n_steps: int = 1000,
        timestep: float = 0.5,
        mass: float = 1.0,
        gamma: float = 0.1,
        save_every: int = 1,
        load_every: int = 1,
        load_forces: bool = False,
        default_atom: str = "N",
        device: str = "cpu",
        preload_sim_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        use_langevin: bool = True,
        initial_positions: np.array = None,
        calculator=None,
    ):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.temperature = temperature
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.timestep = timestep
        self.save_every = save_every
        self.load_every = load_every
        self.default_atom = default_atom
        self.device = device
        self.mass = mass
        self.gamma = gamma
        self.preload_sim_dir = preload_sim_dir
        self.save_path = save_path
        self.use_langevin = use_langevin
        self.initial_positions = initial_positions
        self.calculator = calculator

        self.data = {}

        if preload_sim_dir:
            if isinstance(preload_sim_dir, str):
                preload_sim_dir = Path(preload_sim_dir)
            print("Loading simulations from directory:", preload_sim_dir)
            self.load_simulations(load_forces=load_forces)

            self.n_sims = len(self.data)
            self.n_steps = len(self.data[0]["pos"])
        else:
            print("Running simulations to collect dataest...")
            if self.save_path:
                self.save_path = Path(save_path)
                print(f"Making {self.save_path.as_posix()}")
                self.save_path.mkdir(exist_ok=True)
            self.run_and_save_sims()

        self.all_pos = np.concatenate(
            [self.data[i]["pos"] for i in range(len(self.data))], axis=0
        )

        if load_forces:
            self.all_force = np.concatenate(
                [self.data[i]["force"] for i in range(len(self.data))], axis=0
            )

        if len(self.all_pos.shape) == 2:
            self.all_pos = np.expand_dims(self.all_pos, axis=1)

        self.mean = torch.tensor(np.mean(self.all_pos, axis=0)[:, :2])
        self.std = torch.tensor(np.std(self.all_pos, axis=0)[:, :2])

    def run_and_save_sims(self):
        def update_fu(atom, traj_n):

            new_values = {
                "pos": np.array([atom.get_positions()]),
                "pe": np.array([atom.get_potential_energy()]),
                "force": np.array([atom.get_forces()]),
                "ke": np.array([atom.get_kinetic_energy()]),
            }

            if traj_n not in self.data:
                self.data[traj_n] = new_values
            else:
                for k, v in new_values.items():
                    self.data[traj_n][k] = np.concatenate(
                        (self.data[traj_n][k], v), axis=0
                    )

        if self.initial_positions is not None:
            np.save(
                os.path.join(self.save_path, "initial_positions.npy"),
                self.initial_positions,
            )

        # sample uniformly over initial conditions if provided
        ic_idxs = (
            np.random.choice(len(self.initial_positions), self.n_sims, replace=True)
            if self.initial_positions is not None
            else np.arange(self.n_sims)
        )
        for i, ic_idx in tqdm(enumerate(ic_idxs)):
            if self.initial_positions is not None:
                positions = self.initial_positions[ic_idx].reshape(1, 2)
                positions = np.concatenate([positions, np.zeros((1, 1))], axis=1)
            else:
                # sample uniformly over domain
                if np.random.rand() < 0.5:
                    positions = self.calculator.initial_point
                else:
                    positions = self.calculator.final_point

                x = np.random.uniform(self.calculator.Lx, self.calculator.Hx, (1,))
                y = np.random.uniform(self.calculator.Ly, self.calculator.Hy, (1,))
                positions = np.expand_dims(
                    np.concatenate([x, y, np.zeros((1,))]), axis=0
                )

            atoms = Atoms(
                f"{self.default_atom}",
                positions=positions,
                masses=[self.mass],
            )
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
            atoms.calc = self.calculator

            if self.use_langevin:
                dyn = Langevin(
                    atoms,
                    timestep=self.timestep * units.fs,
                    temperature_K=self.temperature,
                    friction=self.gamma / units.fs,
                )
            else:
                dyn = VelocityVerlet(atoms, timestep=self.timestep * units.fs)

            dyn.attach(update_fu, interval=self.save_every, atom=atoms, traj_n=i)

            if self.save_path:
                save_subdir = (
                    self.save_path
                    / f"temp={self.temperature}_timestep={self.timestep}_friction={self.gamma}"
                )
                save_subdir.mkdir(exist_ok=True, parents=True)
                traj_path = save_subdir / f"mb_{i}.traj"
                traj = Trajectory(traj_path.as_posix(), "w", atoms)
                dyn.attach(traj.write, interval=self.save_every)

            dyn.run(self.n_steps - 1)

    def load_trajectory(self, traj_file, load_forces=False):
        traj = Trajectory(traj_file)
        pos = np.array([a.get_positions() for a in traj[:: self.load_every]])
        if load_forces:
            force = np.array([a.get_forces() for a in traj[:: self.load_every]])
            return {"pos": pos, "force": force}

        return {"pos": pos}

    def load_simulations(self, load_forces=False):
        if isinstance(self.preload_sim_dir, str):
            self.preload_sim_dir = Path(self.preload_sim_dir)
        traj_files = glob.glob(self.preload_sim_dir.as_posix() + "/*.traj")
        for i, traj_file in tqdm(enumerate(traj_files), total=len(traj_files)):
            self.data[i] = self.load_trajectory(traj_file, load_forces=load_forces)

        if os.path.exists(self.preload_sim_dir / "initial_positions.npy"):
            self.initial_positions = np.load(
                self.preload_sim_dir / "initial_positions.npy"
            )

    def __len__(self):
        return len(self.all_pos)

    def __getitem__(self, idx):
        pos = torch.Tensor(self.all_pos[idx][:, :2]).squeeze()
        if hasattr(self, "all_force"):
            force = torch.Tensor(self.all_force[idx][:, :2]).squeeze()
            return pos, force
        return pos


class CustomLangevin:
    """
    Langevin thermostat operating on a batch of MD trajectories in parallel.
    """

    def __init__(self, force_fn, masses, dt, temperature_K, gamma, device):
        self.device = device
        self.force_fn = force_fn
        self.masses = masses
        self.dt = dt * units.fs
        self.temp = temperature_K
        self.temp *= units.kB
        self.gamma = gamma / (1000 * units.fs)
        self.noise_f = (
            torch.tensor(2.0 * self.gamma / self.masses * self.temp * self.dt)
            .sqrt()
            .to(self.device)
        )

    def initialize_velocities(self):

        vel_dist = maxwell()
        velocities = vel_dist.rvs(size=(1, 3))
        # shift so that initial momentum is zero
        velocities -= np.mean(velocities, axis=0)

        # scale velocities to match desired temperature
        sum_vsq = np.sum(np.square(velocities))
        p_dof = 3 * (n_particles - 1)
        correction_factor = math.sqrt(p_dof * self.temp / sum_vsq)
        velocities *= correction_factor
        return torch.Tensor(velocities)

    def step(self, radii, velocities):
        """
        Make a step forward in time with the Langevin integrator.
        Args:
            radii (torch.Tensor): Atom positions (Shape: (N_replicas, N_atoms, 3))
            velocities (torch.Tensor): Atom velocities (Shape: (N_replicas, N_atoms, 3))
            forces (torch.Tensor): Atomic forces (Shape: (N_replicas, N_atoms, 3))
        Returns:
            Updated radii, velocities, forces
        """
        # full step in position
        radii = radii + self.dt * velocities
        # calculate force at new position
        forces = self.force_fn(radii)
        noise = torch.randn_like(velocities).to(forces.device)
        # full step in velocities
        velocities = (
            velocities
            + self.dt * (forces / self.masses - self.gamma * velocities)
            + self.noise_f * noise
        )
        return radii, velocities
