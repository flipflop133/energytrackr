# Energy Consumption Testing Tool

## Overview

This project provides a tool for measuring energy consumption across multiple commits in a Git repository. It automates the process of:

- Cloning the specified repository.
- Checking out each commit sequentially.
- Running a given test command multiple times.
- Measuring energy consumption using a Bash script.
- Detecting significant energy regressions.

## Prerequisites

- Python 3
- Git
- Bash
- Intel RAPL enabled (for power measurement)
- Docker (optional, for running in a container)

## Running the pipeline

### Pre-setup

Before running the pipeline, make sure you have:

- installed the pre-requisites
- ran system_setup.sh to configure the system for energy measurement
- running inside a new tty, without your graphical desktop environment running
- disabled as many processes as possible to avoid interfering with the measurement
- if on laptop, that your laptop is plugged, with the battery fully charged, features like auto-brightness off

### Running system_setup.sh

Run the `system_setup.sh` script to configure the system for energy measurement.

You will have to run it two times if you never ran it.

The first time you'll need to configure your system, this setup only needs to be done one time.

```sh
sudo system_setup.sh first-setup
```

Then reboot your system.

The second time you'll need to configure your system, this setup needs to be done every time.

```sh
sudo system_setup.sh setup
```

As this will put your system into a mode that is suitable for running the pipeline but not for daily use of your computer you can revert the settings in a best effort mode using:

```sh
sudo system_setup.sh revert-first-setup
```

then reboot your system.

you can also revert the setup parameters using :

```sh
sudo system_setup.sh revert-setup
```

which doesn't require a reboot.

### Without Docker

1. Clone the repository

2. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Create a json config file for the project you want to analyze

    The config file must follow the schema defined in [config.schema.json](config.schema.json)

4. Check that the system is stable

    ```sh
    python main.py stability-test
    ```

5. Run the pipeline

    ```sh
    python main.py measure <config_path>
    ```

6. Results

    Results will be produced in a csv file in the same directory as your config file.

7. Analyze results

    Run the plot.py script to visualize the results

    ```sh
    python plot.py <csv_file>
    ```

### With Docker

Follow steps 1 to 3 (included) without docker.

before step 4, run:

    ```sh
    docker buildx build -t pipeline .
    docker run -it --privileged pipeline:latest
    ```

### Running it inside a server with tmux

1. Create a tmux session
    ```sh
    tmux new -s mysession
    ```

2. Reconnect using
    ```sh
    tmux attach -t mysession
    ```

## TODO - sorted by priority

- [x] Don't erase produced CSV files, use a timestamp with project name
- [ ] Automatically detect java version in pom and use export this one, so tests don't fail
- [ ] Build project one time for each commit, for this copy the project x batch times and checkout in each one and compile in each one than for the 30 runs for each commit we just need to run the tests, copying and compiling can be done in parallel and with unlocked frequencies
- [ ] Save run conditions (temperature, CPU governor, number of CPU cycles, etc.), perf could be use for part of this and fastfetch for the rest. Also save config file. Place all this metadata either in the CSV or in a separate file
- [ ] Display run conditions on the graph (e.g. temperature)
- [ ] From measured CPU cycles, use CPU power usage provided by manufacturer to display a second line on the graph to compare energy consumption from RAPL with theoretical estimations.
- [ ] Run only the same tests between commits
- [ ] Do a warm-up run before the actual measurement
- [ ] Add a cooldown between measurements, 1 second by default
- [ ] Check that pc won't go into sleep mode during the test
- [ ] Check that most background processes are disabled
- [ ] Unload drivers/modules that could interfere with the measurement