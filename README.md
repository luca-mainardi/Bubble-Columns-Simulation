# Spherical Harmonics
Bubble columns are extensively used in many industrial applications. High-fidelity numerical simulators are used to simulate bubble columns' spherical harmonics, with great accuracy. The major downside of this method is that they are computationally expensive, sometimes taking multiple days for a single simulation. As a result, there is a large demand for faster surrogate models.

We are looking to build a probabilistic emulator (with parameters $\theta$), that can learn the conditional distribution $P_\theta(X^{t+1}|X^t)$. This emulator should still obtain good accuracy, but much faster than the high-fidelity simulator - thus striking a better balance between accuracy and (time-)complexity.

## File structure

    data/                   - Directory for data files/directories used to train the model(s)

    figures/                - Directory for (evaluation) figures (.png/.svg/.jpg/.gif/.mp4)

    models/                 - Directory for saved model files (.pt) (checkpoints)

    notebooks/              - Directory for Jupyter notebooks (.ipynb)
        utils/                  - - Directory for utility code

    results/                - Directory for results


## Getting Started

### Prerequisites

Make sure you have the following installed:
- Python 3.8 or higher

### Clone the Repository

```sh
git clone git@github.com:luca-mainardi/Bubble-Columns-Simulation.git
cd Bubble-Columns-Simulation
```

### Create a Virtual Environment

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Requirements

```sh
pip install -r requirements.txt
```

### Run Jupyter Notebook

Navigate to the notebook directory and open the desired notebook to get started.

      

