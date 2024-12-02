# CSCE642_CL
# CARLA Setup and Usage Guide

This guide explains how to set up and run the CARLA simulator along with the associated codebase.

---

## Prerequisites

Before getting started, ensure the following are installed on your system:

- **[Conda](https://docs.conda.io/en/latest/miniconda.html)**: For managing the environment.
- **CARLA Simulator**: [Download CARLA](https://carla.org/) and follow the setup instructions.

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

Start the CARLA Simulator
Open a separate terminal and navigate to your CARLA installation directory. Run the simulator:
   ```bash
    ./CarlaUE4.sh
   ```
Running the Main Code
With the CARLA simulator running, execute the main code in the environment:
   ```bash
python main.py
   ```
   
