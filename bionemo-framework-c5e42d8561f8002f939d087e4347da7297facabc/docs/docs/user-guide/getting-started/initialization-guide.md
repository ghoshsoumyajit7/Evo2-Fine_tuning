# Initialization Guide

!!! note

    Prior to beginning this section, you must confirm that your computing platform meets or exceeds the prerequisites
    outlined in the [Hardware and Software Prerequisites](./pre-reqs.md) page and that you have already pulled and
    verified that you can run the BioNeMo container as outlined in the [Access and Startup](./access-startup.md) page.

At this point, you have successfully launched and run the Docker container. This section will guide you through setting
up your host machine environment, suggest Docker commands for various common workflows, and explain helpful `docker run`
options.

## Setting Up Your Host Machine Environment

To effectively use the BioNeMo Framework, we recommend an organized environment configuration and directory
structure. Specifically, we recommend having several cache directories per project. These directories will contain
project files such as data, model checkpoints, training scripts, and outputs such as logs and predictions. To
facilitate container set up, we recommend storing the paths to these directories in a `.env` file that can be referenced
at container runtime. Below, we suggest useful environment variables to define in this file.

### Creating a .env File For First Time Setup

We recommend using a `.env` file in your local workspace to define environment variables.
Specifically, the following variables are useful to include in your `.env` file:

```txt
# Local Cache Directories
LOCAL_RESULTS_PATH
DOCKER_RESULTS_PATH
LOCAL_DATA_PATH
DOCKER_DATA_PATH
LOCAL_MODELS_PATH
DOCKER_MODELS_PATH

# Desired Jupyter Port
JUPYTER_PORT

# NGC Configuration Settings
NGC_CLI_API_KEY
NGC_CLI_ORG
NGC_CLI_TEAM
NGC_CLI_FORMAT_TYPE

# Weights and Biases API Key
WANDB_API_KEY
```

For each of these variables, you can define them inside the `.env` file using `=`. For example, you can set the NGC API
key using `NGC_CLI_API_KEY=<your API key here>`. You can then define these variables in your current shell using:

```bash
source .env
```

Running this command will make these variables available for use in the `docker run` command examples shown below.

!!! note "NGC Credentials Required for Data Download"

    Some of the credentials in the above `.env` file are optional for specific workflows. However, if you intend to use
    data hosted on the NGC platform (for example, model checkpoints and example training data), you _must_ define both
    NGC_CLI_API_KEY and NGC_CLI_ORG at container run time. The easiest way to ensure these variables are set is to use
    the `.env` file as shown here with your specific variable definitions.

Refer to the list below for an explanation of each of these variables:

- `LOCAL_RESULTS_PATH` and `DOCKER_RESULTS_PATH`: Paths for storing results, with `LOCAL` referring to the path on the
    local machine and `DOCKER` referring to the path inside the Docker container.
- `LOCAL_DATA_PATH` and `DOCKER_DATA_PATH`: Paths for storing data, again with `LOCAL` and `DOCKER` distinctions.
- `LOCAL_MODELS_PATH` and `DOCKER_MODELS_PATH`: Paths for storing machine learning models, with the same local and
    Docker differences.
- `JUPYTER_PORT`: The port number for a Jupyter Lab server, default port is 8888.
-  `NGC_CLI_API_KEY`, `NGC_CLI_ORG`, `NGC_CLI_TEAM`, and `NGC_CLI_FORMAT_TYPE`: API key, organization, team, and format
    type for the NVIDIA GPU Cloud (NGC) command-line interface (CLI).
- `WANDB_API_KEY`: An API key for Weights and Biases (W&B), a platform for machine learning experiment tracking and
    visualization.

??? note "Weights and Biases Setup (WANDB_API_KEY, Optional)"

    [Weights and Biases](https://wandb.ai/) (W&B) is a machine learning operations platform that provides tools and
    services to help machine learning practitioners build, train, and deploy models more efficiently. BioNeMo
    is built to work with W&B and requires only simple setup steps to start tracking your experiments. To set up W&B
    inside your container, follow the steps below:

    1. Sign up for an account at [Weights and Biases](https://wandb.ai/).
    2. Setup your [API Key](https://docs.wandb.ai/guides/track/public-api-guide#authentication) with W&B.
    3. Set the `WANDB_API_KEY` variable in your `.env` in the same way as you set the previous environment variable
        above.
    4. Set the environment variable inside your container using the `-e` option, as shown in the next section.

## Starting the BioNeMo Container for Common Workflows

Below we describe some common BioNeMo workflows, including how to setup and run the container in each case. Each of the
following examples will assume that you have local workspace directories as defined in your `.env` file shown above that
you will attach to the container via volume mounts.

### Starting a Shell Inside the Container

With a shell inside the BioNeMo Docker container, you can execute commands, edit files, and run applications as if you
were working directly on the host machine. This self-contained environment allows you to work with your project's
dependencies and configurations in isolation, ensuring consistent results and reproducibility. You can install packages,
test and debug applications, and customize the environment to suit your needs.

You can launch a Bash shell inside the BioNeMo container using the command below. Note that any files modified in the
mounted directories while inside the container will persist on the host machine, but other modifications (such
as installed software) will not.

```bash
docker run \
  --rm -it \
  --gpus all \
  --network host \
  --shm-size=4g \
  -e WANDB_API_KEY \
  -e NGC_CLI_API_KEY \
  -e NGC_CLI_ORG \
  -e NGC_CLI_TEAM \
  -e NGC_CLI_FORMAT_TYPE \
  -v $LOCAL_DATA_PATH:$DOCKER_DATA_PATH \
  -v $LOCAL_MODELS_PATH:$DOCKER_MODELS_PATH \
  -v $LOCAL_RESULTS_PATH:$DOCKER_RESULTS_PATH \
  {{ docker_url }}:{{ docker_tag }} \
  /bin/bash
```

* `--rm`: Removes the container when it exits.
* `-it`: Allocates a pseudo-TTY and keeps the container running in the foreground.
* `--gpus all`: Allocates all available GPUs on the host machine.
* `--network host`: Allows the container to use the host's network stack, effectively sharing the host's network
    namespace and allowing the container to access the host's network interfaces directly.
* `--shm-size=4g`: Sets the size of the shared memory (/dev/shm) in the container to 4 gigabytes, which can be useful for applications that rely heavily on shared memory.
* `-e <VARIABLE>`: Sets the environment variable inside the container, taking the value set on the host machine.
* `-v <LOCAL DIRECTORY>:<DOCKER DIRECTORY>`: Mounts a volume from the host machine to the container.
* `{{ docker_url }}:{{ docker_tag }}`: The path to the Docker image to use.
* `/bin/bash`: The command to run inside the container, which starts a Bash shell.

### Running a Model Training Script Inside the Container

Running a model training script inside the BioNeMo Docker container is the preferred workflow for model training. The
container provides an encapsulated and reproducible training environment. By mounting a volume from the host machine,
the output directory containing results such as logs and checkpoints can be persisted even after the container is
removed. A training script can be run as in the example below. Replace `training.py` and option (for example,
`--option1`) with the file name and relevant command line options, respectively.

```
docker run --rm -it --gpus all \
  -e NGC_CLI_API_KEY \
  -e WANDB_API_KEY \
  -v $LOCAL_DATA_PATH:$DOCKER_DATA_PATH \
  -v $LOCAL_MODELS_PATH:$DOCKER_MODELS_PATH \
  -v $LOCAL_RESULTS_PATH:$DOCKER_RESULTS_PATH \
  {{ docker_url }}:{{ docker_tag }} \
  python $DOCKER_RESULTS_PATH/training.py --option1 --option2 --output=$DOCKER_RESULTS_PATH
```

Many of the Docker run options are identical to the shell example above, with the exception of the command being run:

* `python $DOCKER_RESULTS_PATH/training.py --option1 --option2 --output=$DOCKER_RESULTS_PATH`: The command to run inside the
container, which runs the `training.py` Python script with the specified command-line arguments.

### Running Jupyter Lab Inside the Container

By starting a Jupyter Lab instance inside the BioNeMo Framework container, users can leverage the container's optimized
environment for machine learning workloads to accelerate their data science workflows, while also benefiting from the
interactive and collaborative features of Jupyter Lab. This allows users to seamlessly transition between data
preparation, model development, and visualization, all within a single, streamlined environment. You can then launch the
container. We recommend running the container in a Jupyter Lab environment using the command below:

```bash
docker run --rm -d --gpus all \
  -p $JUPYTER_PORT:$JUPYTER_PORT \
  -e NGC_CLI_API_KEY \
  -e WANDB_API_KEY \
  -v $LOCAL_DATA_PATH:$DOCKER_DATA_PATH \
  -v $LOCAL_MODELS_PATH:$DOCKER_MODELS_PATH \
  -v $LOCAL_RESULTS_PATH:$DOCKER_RESULTS_PATH \
  {{ docker_url }}:{{ docker_tag }} \
  jupyter lab \
  	--allow-root \
	--ip=* \
	--port=$JUPYTER_PORT \
	--no-browser \
  	--NotebookApp.token='' \
  	--NotebookApp.allow_origin='*' \
  	--ContentsManager.allow_hidden=True \
  	--notebook-dir=$DOCKER_RESULTS_PATH
```

Refer to the guide below for an explanation of the recommended Jupyter Lab options:

* `jupyter lab ...`: The command to run inside the container, which starts a Jupyter Lab server. The options are:
	+ `--allow-root`: Allow the Jupyter Lab server to run as the root user.
	+ `--ip=*`: Listen on all available network interfaces, which allows access from outside the container.
	+ `--port=$JUPYTER_PORT`: Listen on port 8888.
	+ `--no-browser`: Do not open a browser window automatically.
	+ `--NotebookApp.token=''`: Set an empty token for the Jupyter Lab server (no authentication is required).
	+ `--NotebookApp.allow_origin='*'`: Allow requests from any origin.
	+ `--ContentsManager.allow_hidden=True`: Allow the contents manager to access hidden files and directories.
	+ `--notebook-dir=$DOCKER_RESULTS_PATH`: Set the notebook directory to
        `$DOCKER_RESULTS_PATH` inside the container.

## Common `docker run` Options

Below we explain some common `docker run` options and how to use them as part of your BioNeMo development workflows.

### Mounting Volumes with the `-v` Option

The `-v`  allows you to mount a host machine's directory as a volume inside the
container. This enables data persistence even after the container is deleted or restarted. In the context of machine
learning workflows, leveraging the `-v` option is essential for maintaining a local cache of datasets, model weights, and
results on the host machine such that they can persist after the container terminates and be reused across container
runs.

**Syntax:**

```
docker run -v <host_directory>:<container_directory> <image_name>
```
**Example:**

```
docker run -v /path/to/local/cache:/workspace/bionemo2/cache \
    {{ docker_url }}:{{ docker_tag }}
```

In this example, the `/path/to/local/cache` directory on the host machine is mounted as a volume at
`/workspace/bionemo2/cache` inside the container.

### Setting Environment Variables with the `-e` Option

The `-e` option allows you to set environment variables inside the container. You can use this option to define
variables that will be available to the application running inside the container.

**Example:**

```bash
docker run -e MY_VAR=value -e ANOTHER_VAR=another_value \
    {{ docker_url }}:{{ docker_tag }}
```

- `-e MY_VAR=value` sets the `MY_VAR` environment variable to `value` inside the container.
- `-e ANOTHER_VAR=another_value` sets the `ANOTHER_VAR` environment variable to `another_value` inside the container.

You can set multiple environment variables by repeating the `-e` option. The values of these variables will be available
to the application running inside the container, allowing you to customize its behavior.

Note that you can also use shell variables and command substitutions to set environment variables dynamically. For
example:

```bash
MY_EXTERNAL_VAR=external_value
docker run -e MY_INTERNAL_VAR=$MY_EXTERNAL_VAR \
    {{ docker_url }}:{{ docker_tag }}
```

In this example, the `MY_INTERNAL_VAR` environment variable inside the container will be set to the value of the
`MY_EXTERNAL_VAR` shell variable on the host machine.

### Setting User and Group IDs with the `-u` Option

The `-u` option sets the user and group IDs to use for the container process. By matching the IDs of the user on the
host machine, the user inside the container will have identical permissions for reading and writing files in the mounted
volumes as the user that ran the command. You can use command substitutions to automatically retrieve your user and
group IDs.

**Example:**

```bash
docker run -u $(id -u):$(id -g) \
    {{ docker_url }}:{{ docker_tag }}
```

- `$(id -u)` is a command substitution that executes the id -u command and captures its output. `id -u` prints the
    effective user ID of the current user.
- `$(id -g)` is another command substitution that executes the `id -g` command and captures its output. `id -g` prints
    the effective group ID of the current user.
