import importlib
import logging
import shutil
import subprocess
import typing as t

import jinja2
import numpy as np
import xarray as xr

from netCDF4 import Dataset

from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
Input = namedtuple("Input", ["name", "source", "target", "needs_jinja"])
ForcingField = namedtuple("ForcingField", ["name", "values", "attrs"])
CaseLayout = namedtuple("CaseStructure", ["base", "expref", "inputs"])
pathlike = t.Union[str, Path]

casename = lambda dx: f"UNAGI_R{int(dx*1e-3):03d}"
filename = lambda dx, suffix: f"{casename(dx)}_{suffix}.nc"


@dataclass
class UnagiDomain:
    """Describes the domain of the UNAGI case

    :param dx: The spacing of the grid in the x-direction [m]
    :type dx: float
    :param dy: The spacing of the grid in the y-direction [m]
    :type dy: float
    :param Lx: The length of the domain in the x-direction [m]
    :type Lx: float, optional
    :param Ly: The length of the domain in the y-direction [m]
    :type Ly: float, optional
    :param depth: The depth of the domain [m]
    :type depth: float, optional
    :param beta_lat: The latitude to calculate the value of beta [deg]
    :type depth: float, optional
    """

    dx: float
    dy: float = None
    Lx: float = 9000 * 1.0e3
    Ly: float = 2400 * 1.0e3
    depth: float = 3000.0
    beta_lat: float = -50.0

    @property
    def dims(self):
        return self.nz, self.ny, self.nx

    def __post_init__(self):
        """Derived quantities associated with the grid"""

        self.dy = self.dx if self.dy is None else self.dy
        self.nz = 31  # + 1 for the bottom
        self.nx = int(self.Lx // self.dx)
        self.ny = int(self.Ly // self.dy + 2)  # Walls

        # Offset to ensure that U-point is at the peak of ridge
        self.x = self.dx * (np.arange(0, self.nx) - 0.5)
        # Offset to ensure that bottom ocean-row is at 0.
        self.y = self.dy * (np.arange(0, self.ny) - 1.0)
        self.nav_lon, self.nav_lat = np.meshgrid(self.x, self.y)
        self.bathymetry = None
        self.file_suffix = "bathymetry"

    def generate_bathymetry(self, ridge_H=1500.0, ridge_L=500.0 * 1e3) -> xr.DataArray:
        """Generate the bathmetry for the UNAGI case (subsurface meridional ridge)

        :param ridge_H: Height of the ridge [m], defaults to 1500m
        :param ridge_L: Width of the ridge [m], defaults to 500000m

        """
        ridge_center = 0.5 * self.Lx
        ridge_left = ridge_center - ridge_L
        ridge_right = ridge_center + ridge_L

        # Modify the topography to insert a cosine ridge
        ridge_slice = np.zeros(len(self.x)) + self.depth
        ridge_idx = (self.x >= ridge_left) & (self.x <= ridge_right)
        ridge_slice[ridge_idx] = ridge_slice[ridge_idx] - (
            (0.5 * ridge_H)
            * (1 + np.cos(np.pi * (self.x[ridge_idx] - ridge_center) / ridge_L))
        )
        # Add walls to the top and bottom of the channel
        bathymetry = np.zeros((self.ny, self.nx))
        bathymetry[:, :] = ridge_slice
        bathymetry[0, :] = 0.0
        bathymetry[-1, :] = 0.0
        bathymetry = bathymetry[np.newaxis, :, :]

        # Make this an xarray dataset to help in visualization and writing
        attrs = {"units": "m"}
        coords = {
            "nav_lon": ("x", self.x, attrs),
            "nav_lat": ("y", self.y, attrs),
        }
        self.bathymetry = xr.DataArray(
            bathymetry,
            name="bathymetry",
            dims=("t", "y", "x"),
            coords=coords,
            attrs=attrs,
        )
        return self.bathymetry

    def write_bathymetry(self, domain_dir: pathlike) -> Path:
        domain_path = Path(domain_dir)
        domain_path.mkdir(parents=True, exist_ok=True)
        if self.bathymetry is not None:
            self.bathymetry_path = domain_path / filename(self.dx, "bathymetry")
            self.bathymetry.to_netcdf(self.bathymetry_path, unlimited_dims="t")
        else:
            raise Exception(
                "Bathymetry has not been created. Use method `generate_bathymetry` "
                + "before calling `write_bathymetry`"
            )

    def write_domain(
        self,
        domaincfg_exe: pathlike,
        domain_dir: pathlike,
    ) -> Path:

        domain_path = Path(domain_dir)

        with importlib.resources.open_text(
            "nemoecwg.cases.unagi.inputs", "domaincfg_cfg.jinja"
        ) as f:
            domain_cfg_template = jinja2.Template(f.read())

        with open(domain_path / "namelist_cfg", "w") as f:
            f.write(
                domain_cfg_template.render(
                    {
                        "cn_topo": self.bathymetry_path.name,
                        "dx": self.dx,
                        "dy": self.dy,
                        "ni": self.nx,
                        "nj": self.ny,
                        "nk": self.nz,
                        "depth": self.depth,
                        "lam0": self.x[0],
                        "phi0": self.y[0],
                    }
                )
            )
        namelist_ref = importlib.resources.files(
            "nemoecwg.cases.unagi.inputs"
        ).joinpath("domaincfg_ref")
        shutil.copy(namelist_ref, domain_path / "namelist_ref")

        subprocess.check_call(domaincfg_exe, cwd=domain_dir)
        outpath = domain_path / filename(self.dx, "domain")
        (domain_path / "domain_cfg.nc").replace(outpath)
        self.domain_path = outpath
        self.ds = xr.open_dataset(self.domain_path)
        return self.ds


@dataclass
class UnagiForcing:
    """Container for generating forcing for the UNAGI case

    :param domain: An initialized UNAGI domain
    :type domain: UnagiDomain
    :param tau0: The maximum windstress in the centre of the domain [N m-2]
    :type tau0: float
    """

    domain: UnagiDomain
    tau0: float = 0.2

    def __post_init__(self):
        # Note: NEMO's always assumes all fluxes are on the T-point
        zero = np.zeros((1, self.domain.ny, self.domain.nx))
        units = {
            "utau": "N m-2",
            "vtau": "N m-2",
            "qtot": "W m-2",
            "qsr": "W m-2",
            "emp": "m s-1",
        }
        forcing_fields = {
            var: ForcingField(var, zero, {"units": units})
            for var, units in units.items()
            if var != "utau"
        }
        # Generate utau
        y_mid = 0.5 * (self.domain.y[0] + self.domain.y[-1])
        y_scaled = (self.domain.nav_lat - y_mid) / self.domain.Ly
        jet = lambda y: (0.5 * self.tau0) * (1 + np.cos(2 * np.pi * y))
        forcing_fields["utau"] = ForcingField(
            "utau", jet(y_scaled)[np.newaxis, ...], {"units": units["utau"]}
        )

        # Create an xarray dataset with all the forcing
        dims_2d = ("y", "x")
        dims_3d = ("t", "y", "x")
        coords = {
            "nav_lon": (dims_2d, self.domain.nav_lon),
            "nav_lat": (dims_2d, self.domain.nav_lat),
        }
        self.ds = xr.Dataset(coords=coords)
        for forcing in forcing_fields.values():
            da = xr.DataArray(
                forcing.values,
                name=forcing.name,
                dims=dims_3d,
                coords=coords,
                attrs=forcing.attrs,
            )
            self.ds[forcing.name] = da

    def write(self, forcing_dir):
        self.path = Path(forcing_dir) / filename(self.domain.dx, "forcing")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ds.to_netcdf(self.path)


@dataclass
class UnagiInitialCondition:
    """Initial conditions for the UNAGI case

    :param domain: An initialized UNAGI domain
    :type domain: UnagiDomain
    :param dtheta: Magnitude of the temperature difference across the domain[deg C]
    :type dtheta: float
    :param theta_noise: Magnitude of the random noise added to the temperature [deg C]
    :type theta_noise: float
    :param theta_min: Minimum temperature allowed in the domain [deg C]
    :type theta_min: float
    :param z0": The e-folding scale for the stratification [m]
    :type z0: float
    :param s0": The (constant) salinity in the domain [ppt]
    :type s0: float
    """

    domain: UnagiDomain
    dtheta: float = 15.0
    theta_noise: float = 0.05
    theta_min: float = 0.25
    z0: float = 1000.0
    s0: float = 35.0

    def __post_init__(self):
        nt = 1  # Dummy time level
        nz, ny, nx = self.domain.dims
        soce = self.s0 + np.zeros((nz, ny, nx))
        sss = self.s0 + np.zeros((nt, ny, nx))

        ygrid, zgrid = np.meshgrid(self.domain.y, -self.domain.ds.nav_lev)
        toce = np.zeros((nz, ny, nx))
        t_slice = self.dtheta * (
            (ygrid / self.domain.Ly)
            * (np.exp(zgrid / self.z0) - np.exp(-self.domain.depth / self.z0))
            / (1.0 - np.exp(-self.domain.depth / self.z0))
        )
        toce[:,:,:] = t_slice[..., np.newaxis]

        # Remove a small offset due to the vertical coordinate
        toce -= toce[0, 0, 0]
        toce += np.random.normal(0.0, self.theta_noise, toce.shape)
        toce[toce < self.theta_min] = self.theta_min
        sst = toce[0, :, :]
        sst = sst[np.newaxis, ...] # Add a time dimension

        # Create the xarray dataset
        coords = self.domain.ds.coords.copy()
        dims_2d = ("y", "x")
        coords.update(
            {
                "nav_lon": (dims_2d, self.domain.nav_lon),
                "nav_lat": (dims_2d, self.domain.nav_lat),
            }
        )
        self.ds = xr.Dataset(coords=self.domain.ds.coords)
        make_da = lambda values, name, dims, attrs: xr.DataArray(
            values, name=name, dims=dims, attrs=attrs
        )
        ts_dims = ("nav_lev", "nav_lat", "nav_lon")
        surf_dims = ("t", "y", "x")

        s_attrs = {"units": "g kg-1"}
        self.ds["soce"] = make_da(soce, "soce", ts_dims, s_attrs)
        self.ds["sss"] = make_da(sss, "sss", surf_dims, s_attrs)

        t_attrs = {"units": "C"}
        self.ds["toce"] = make_da(toce, "toce", ts_dims, t_attrs)
        self.ds["sst"] = make_da(sst, "sst", surf_dims, t_attrs)

    def write(self, state_dir):
        self.path = Path(state_dir) / filename(self.domain.dx, "initial_state")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ds.to_netcdf(self.path)


@dataclass
class UnagiCase:
    """Creates the actual case directory for UNAGI"""

    domain: UnagiDomain
    forcing: UnagiForcing
    initial_conditions: UnagiInitialCondition
    outpath: pathlike
    nemo_components: tuple[str] = ("OCE",)

    def __post_init__(self):
        self.name = casename(self.domain.dx)
        base = Path(self.outpath) / self.name
        self.case_dirs = CaseLayout(base, base / "EXPREF", base / "INPUTS")
        fstore = importlib.resources.files("nemoecwg.cases.unagi.inputs")

        # cpp file needs to be renamed
        cpp_target = self.case_dirs.base / f"cpp_{self.name}.fcm"

        self.inputs = [
            Input("forcing", self.forcing.path, self.case_dirs.inputs, False),
            Input("domain", self.domain.domain_path, self.case_dirs.inputs, False),
            Input("ICs", self.initial_conditions.path, self.case_dirs.inputs, False),
            Input("cpp", fstore.joinpath("cpp_UNAGI.fcm"), cpp_target, False),
            Input(
                "EXPREF_DIR", fstore.joinpath("EXPREF"), self.case_dirs.expref, False
            ),
            Input(
                "namelist_cfg",
                fstore.joinpath("namelist_cfg.jinja"),
                self.case_dirs.expref,
                True,
            ),
        ]

        self.jinja_vars = {
            "casename": casename(self.domain.dx),
            "domain_file": self.domain.domain_path.stem,
            "state_file": self.initial_conditions.path.stem,
            "forcing_file": self.forcing.path.stem,
        }

    def _process_jinja(self, input: Input):
        with open(input.source, "r") as f:
            template = jinja2.Template(f.read())
        with open(input.target / input.source.stem, "w") as f:
            f.write(template.render(self.jinja_vars))

    @staticmethod
    def _process_directory(input: Input):
        for item in input.source.iterdir():
            shutil.copy(item, input.target)

    def generate_case_directory(self):
        for d in self.case_dirs:
            d.mkdir(parents=True, exist_ok=True)

        for input in self.inputs:
            if input.needs_jinja:
                self._process_jinja(input)
            elif input.source.is_dir():
                self._process_directory(input)
            elif input.source.is_file():
                shutil.copy(input.source, input.target)

        print( f"Case {self.name} created at {self.case_dirs.base.resolve()}.")


    def _modify_ref_cfgs(self, ref_cfgs_file):
        with open(ref_cfgs_file, "r") as f:
            lines = f.readlines()

        found = False
        entry = f"{self.name} {' '.join(self.nemo_components)}\n"
        with open(ref_cfgs_file, "w") as f:
            for line in lines:
                if line.startswith(self.name):
                    f.write(entry)
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(entry)

    def install(self, nemo_cfg_dir: pathlike, force=False):
        nemo_cfg_path = Path(nemo_cfg_dir)
        target = nemo_cfg_path / self.name

        if target.exists() and not force:
            raise Exception(
                f"{self.name} already exists in {nemo_cfg_dir}. "
                + "Use force=True to overwrite."
            )
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.case_dirs.base, nemo_cfg_path/self.name)

        ref_cfgs_file = nemo_cfg_path / "ref_cfgs.txt"
        self._modify_ref_cfgs(ref_cfgs_file)



