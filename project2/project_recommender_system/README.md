# Project Recommender System

## Team members

* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

## Usage

### Docker

We provide a Docker file which allows the project to be run directly. It is
based on [gw0/docker-keras](https://github.com/gw0/docker-keras).

#### Running

```bash
 docker run -it dtsbourg/cs433-project:latest bash
```

The default image allows for exploration of the project from within the image.
The `WORKDIR` is the base of the project. See the [Structure](#) section for more
information on how the repository is organised.

You can then build the project as follows :

```bash
cd src
python3 run.py
```

By default the `main.py` script will launch both the training and prediction
pipelines. To run only the prediction (by default with our best model) or only
the training (on our best performing model), please run :

```bash
cd src
python3 run.py --train # for training, or
python3 run.py --predict # for prediction
```

#### building

The image can be rebuilt if necessary by running :

```bash
docker build -t cs433-project . -f Dockerfile # Exploring
```
> Warning : The datasets are loaded in memory so please make sure the allocated
memory is sufficient. This can be configured in Docker's preferences.

### Running locally

Start by installing the dependencies if necessary :

```bash
sudo python3 -m pip install -r requirements.txt
```

You can then run, train or predict with the model by using one of
the following commands. **The data has to be unzipped first
so the model can load it.**

```bash
cd data
tar -zxvf data.tgz
cd embeddings
tar -zxvf embeddings.tgz
cd ../../src
python3 run.py # for the full pipeline, or
python3 run.py --train # for training, or
python3 run.py --predict # for prediction
```  

## Structure

### Repository

In tree form, the repository has the following structure.

```bash
.
├── Dockerfile
├── README.md
├── data                    # Datasets
│   ├── data.tgz            # Given data
│   ├── embeddings          # Pre-computed embeddings
│       └── embeddings.tgz
├── notebook                # For data exploration
│   ├── Exploration.ipynb
│   └── NN.ipynb
├── notes.md               # Internal project notes
├── requirements.txt       # python libraries
├── res                    # Results of our model
│   ├── img                # Loss plots
│   │   └── ....png
│   ├── model              # Model checkpoints 
│   │   ├── Dense_Final
│   │   ├── Shallow_Final
│   │   └── best
│   └── pred               # Model predictions
│       └── submission_best.csv
└── src
    ├── baselines.py
    ├── blending.py
    ├── data.py
    ├── exploration.py
    ├── model_graveyard.py
    ├── models.py
    ├── pipe.py
    ├── pipeline.py
    ├── run.py
    ├── surprise_lib.py
    └── utils.py
```

### `src`

* `baselines.py` : interface for running the baselines.
* `blending.py` : interface for running the blending procedure
* `data.py` : main interface for interacting with the dataset. Defines a set of helper functions to load and manipulate the data.
* `exploration.py` : interface for some of our early exploratory work, including figures for the report.
* `model_graveyard.py` : A collection of experimental, partial, unused or deprecated models.
* `models.py` : main model definition module.
* `pipe.py` : interface for heavy lifting.
* `pipeline.py` : Defines some common pipeline used in this project.
* `run.py` : main interface.
* `surprise_lib.py` : interface for running the baselines with [Suprise](http://surpriselib.com/).
* `utils.py` : Defining several useful utilities.

## Report


