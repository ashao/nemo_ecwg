Quick Start
-----------

1. Clone this repository
```
git clone https://github.com/ashao/nemo_ecwg.git
```

1. Install this python package
```
cd nemo_ecwg
pip install -e .
```

1. Change `DOMAINCFG_EXE` and `NEMO_CFG_DIR` variables in
`examples/unagi.ipynb` and run the notebook

1. Go to your NEMO source directory (where `makenemo` is) and compile
```
cd PATH_TO_NEMO_SOURCE
./makenemo -m YOUR_ARCH -r UNAGI_R100 -j 8
```

1. Copy the inputs into the `EXP00` directory
```
cd cfgs/UNAGI_R100
cp INPUTS/* EXP00
```

1. Run the UNAGI case
```
cd EXP00
mpirun -n 8 ./nemo
```