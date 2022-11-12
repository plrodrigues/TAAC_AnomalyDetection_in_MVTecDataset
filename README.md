# TAAC: AnomalyDetection using the MVTecDataset


## Project structure

- [/data](/data) - contains the data used on this project 
- [/results](/results/) - contains images and other things for reporting
- [/src](/src) - contains the main code
    - [/commons](/src/commons/) - contains the code that can be reused independently on the model selected, e.g. datasets, dataloaders, metrics
    - [/models](/src/models/) - contains the code of the different models to solve this anomaly detection and segmentation problem
- [/notebooks](/notebooks) -  directory that holds all the exploratory notebooks


## Dataset

[MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)


## Conda environment

- create the conda environment with the packages defined in 8[equirements.txt](/requirements.txt)
```sh
python -m venv venv_taac .
```

- activate the environment generated
```sh
venv_taac
```

- install pachages
```sh
python -m pip install -r requirements.txt
```

- deactivate the environment
```sh
conda deactivate
```