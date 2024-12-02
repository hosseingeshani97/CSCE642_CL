# CSCE642_CL
# CARLA Setup and Usage Guide

This guide explains how to set up and run the CARLA simulator along with the associated codebase.

---

## Prerequisites

Before getting started, ensure the following are installed on your system:

- **[Conda](https://docs.conda.io/en/latest/miniconda.html)**: For managing the environment.
- **CARLA Simulator**: [Download CARLA](https://github.com/carla-simulator/carla/blob/master/Docs/download.md) and follow the setup instructions. We downloaded Carla0.9.15
- Using Python 3.9
- Using Windows 11

---

## Environment Setup

1. **Create and Activate Conda Environment**

   Use the provided `environment.yml` file to create the required Conda environment. Run the following command:

   ```bash
   conda env create -f environment.yml -n CL_env
   ```
   Activate the environment:
   ```bash
   conda activate CL_env
   ```

2. **Start the CARLA Simulator**

   Open a separate anaconda prompt terminal and navigate to your CARLA installation directory. Run the simulator:
   ```bash
    CarlaUE4
   ```
3. **Running the Main Code**
   With the CARLA simulator running, in our VS code terminal with the same python interpreter/environment activated, execute the main code in the environment:
   ```bash
   python PPO_CNN_DiscreteVersion375Percent.py --s 10 --epsilonInitial 1 --discount .15 --throttle .5 --preview true --numEpisodes 1 -l y -q 20000
   ```
   If you want Curriculum learning activated, change the curriculumLearning = False (line 236) to True
   
