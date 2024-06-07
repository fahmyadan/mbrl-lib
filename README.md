[![PyPi Version](https://img.shields.io/pypi/v/mbrl)](https://pypi.org/project/mbrl/)
[![Main](https://github.com/facebookresearch/mbrl-lib/workflows/CI/badge.svg)](https://github.com/facebookresearch/mbrl-lib/actions?query=workflow%3ACI)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/mbrl-lib/tree/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
 
 # Planet Highway-Env

 Run the shell script to install dependencies. 
 Project requires Python >=3.10, conda and torch with cuda 12.

 ``` bash 
 sudo chmod +x setup.bash 
./setup.bash 
conda activate mbrl
python3 mbrl/examples/main.py algorithm=planet overrides=planet_highway_env dynamics_model=planet_hw action_optimizer=mppi
 ```
to run the ensembles: `git checkout dev/ensemble`. The instructions are there. 

There are four main config files that must be selected to run the main file. The corresponding folders can be found in `mbrl/examples/conf`. At the moment, the following configurations for the intersection problem have been tested:

`algorithm=planet overrides=planet_highway_env dynamics_model=planet_hw`

The desired action optimizer is MPPI.

The main configs (e.g. wandb, env_args etc.) can be found in `overrides=planet_highway_env`


# MBRL-Lib

``mbrl`` is a toolbox for facilitating development of 
Model-Based Reinforcement Learning algorithms. It provides easily interchangeable 
modeling and planning components, and a set of utility functions that allow writing
model-based RL algorithms with only a few lines of code. 

See also our companion [paper](https://arxiv.org/abs/2104.10159). 

### Installation

## Provided algorithm implementations
MBRL-Lib provides implementations of popular MBRL algorithms 
as examples of how to use this library. You can find them in the 
[mbrl/algorithms](https://github.com/facebookresearch/mbrl-lib/tree/main/mbrl/algorithms) folder. Currently, we have implemented
[PETS](https://github.com/facebookresearch/mbrl-lib/tree/main/mbrl/algorithms/pets.py),
[MBPO](https://github.com/facebookresearch/mbrl-lib/tree/main/mbrl/algorithms/mbpo.py),
[PlaNet](https://github.com/facebookresearch/mbrl-lib/tree/main/mbrl/algorithms/planet.py), 
we plan to keep increasing this list in the future.

The implementations rely on [Hydra](https://github.com/facebookresearch/hydra) 
to handle configuration. You can see the configuration files in 
[this](https://github.com/facebookresearch/mbrl-lib/tree/main/mbrl/examples/conf) 
folder. 
The [overrides](https://github.com/facebookresearch/mbrl-lib/tree/main/mbrl/examples/conf/overrides) 
subfolder contains
environment specific configurations for each environment, overriding the 
default configurations with the best hyperparameter values we have found so far 
for each combination of algorithm and environment. You can run training
by passing the desired override option via command line. 
For example, to run MBPO on the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/) version of HalfCheetah, you should call
```python
python -m mbrl.examples.main algorithm=mbpo overrides=mbpo_halfcheetah 
```
By default, all algorithms will save results in a csv file called `results.csv`,
inside a folder whose path looks like 
`./exp/mbpo/default/gym___HalfCheetah-v2/yyyy.mm.dd/hhmmss`; 
you can change the root directory (`./exp`) by passing 
`root_dir=path-to-your-dir`, and the experiment sub-folder (`default`) by
passing `experiment=your-name`. The logger will also save a file called 
`model_train.csv` with training information for the dynamics model.

Beyond the override defaults, You can also change other configuration options, 
such as the type of dynamics model 
(e.g., `dynamics_model=basic_ensemble`), or the number of models in the ensemble 
(e.g., `dynamics_model.model.ensemble_size=some-number`). To learn more about
all the available options, take a look at the provided 
[configuration files](https://github.com/facebookresearch/mbrl-lib/tree/main/mbrl/examples/conf). 

## License
`mbrl` is released under the MIT license. See [LICENSE](LICENSE) for 
additional details about it. See also our 
[Terms of Use](https://opensource.facebook.com/legal/terms) and 
[Privacy Policy](https://opensource.facebook.com/legal/privacy).

## Citing
If you use this project in your research, please cite:

```BibTeX
@Article{Pineda2021MBRL,
  author  = {Luis Pineda and Brandon Amos and Amy Zhang and Nathan O. Lambert and Roberto Calandra},
  journal = {Arxiv},
  title   = {MBRL-Lib: A Modular Library for Model-based Reinforcement Learning},
  year    = {2021},
  url     = {https://arxiv.org/abs/2104.10159},
}
```
