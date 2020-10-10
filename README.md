# Falsify Neural Network Controlled F16 GCAS

## Structure
```sh
├── F16_falsify
│   ├── demo.ipynb          # Notebook Demo
│   ├── falsify
│   │   ├── bo.py           # Falsify algorithm
│   │   └── para.py         # Paralization Tool
│   ├── pretrained          # Pretrained Neural Network
│   │   ├── ddpg.pkl
│   │   └── ppo.zip
│   ├── results
│   └── utils
│       ├── loader.py       # Neural Network Loader
│       └── simulation.py   # Simulation functions
├── README.md
└── setup.py
```

## Demo
Passed:    
![Passed](F16_falsify/results/f16_passed.gif)

Failed:   
![Failed](F16_falsify/results/f16_unsafe.gif)

See [Demo Notebook](F16_falsify/demo.ipynb).