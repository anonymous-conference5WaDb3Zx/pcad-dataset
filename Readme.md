## Data
All data is available in cloud storage with the following link: https://drive.google.com/drive/folders/13ryvR4XNhJHqDAsKGM_QEkrHOTWrj_7f?usp=sharing

* .devcontainer: aux files to build the docker container from scratch
* pcad-dataset-raw.tar: raw data and auxiliary files
* pcad-dataset-runner.tar: docker image to run on source code
* pcad-dataset-final.zip: Final dataset as result from running the dataset generation pipeline
* pcad-dataset-final-lighting-variations.zip: Final dataset for variations in lighting
* pcad-dataset-final-synthetic: Final dataset with synthetic data

## Installation

### Prerequisits
For the code to run, you need to start a docker container. The authors use WSL for that, but any popular native linux distribution should also work. For details about the installation, visit https://docs.docker.com/desktop/wsl/
The docker container has blender installed which makes use of the dedicated nvidia gpu, if available. The code runs also with cpu only but probably much slower. Make sure that your nvidia gpu is visible to the docker daemon.
The pipeline will generate intermediate files that take up approxemately 50GB. The final dataset itself takes up space for 

### Installation of data
pcad-dataset-data contains all data needed to generate the final dataset. Please make sure that pcad-dataset-raw is inflated into the project root as `data/`.

### Using docker
The pre-built docker image pcad-dataset-runner.tar can be extracted and run using the command

`docker load < pcad-dataset-runner.tar`

Check if the container has been loaded successfully:

`docker image list`

It should list the docker container as "pcad-dataset-runner"

### Adapt run.sh
The container is started according to the scheme of Dev Containers. We simulate this behavior in the run-script that can be found in .dev-container/run.sh by attaching the code as volume into the runinng container. In here, please change the placeholder \<repo location\> with the location of the repository you pulled earlier. The path /home/mambauser/dev-container remains unchanged.

### Running dataset generation
In our code, we make heavy use of the framework Data Versioning Control (DVC) to model the reproducible dataset generation pipeline. It is sufficient to run the command

`dvc repro`

inside the container's bash shell to generate results in results/own. The time needed to generate the whole datasets depends on the hardware. It should be expected to take several hours.

You may try to run the python script
`python -m src.runner`
that enables running multiple dvc stages in parallel. However, this is not officically supported by dvc and should be used with caution. Furthermore, the parallelization will potentially use a lot of resources. If you want to try it out, edit the lines 91 to 134, manually to run certain dvc stages.

After the pipeline has been run, the dataset can be found in `results/own`.

If you want to generate results for the dataset with variations in lighting, the directory needs to be renamed `data/images`. By default `data/images` contains the dataset without variations in lighting.
