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
python3 main.py
```

By default the `main.py` script will launch both the training and prediction
pipelines. To run only the prediction (by default with our best model) or only
the training (on our best performing model), please run :

```bash
cd src
python3 main.py --train # for training, or
python3 main.py --predict # for prediction
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
python3 neural.py # for the full pipeline, or
python3 main.py --train # for training, or
python3 main.py --predict # for prediction
```  

## Structure

### `.`
The TLD contains :
* `README.md`
* `notes.md` some working notes on the project used within the group
* `Dockerfile`
* `requirements.txt` specifying the required packages
* `.gitignore`

### `/src/`

Contains all the logic for our project.

### `/res/`

Contains the outputs of the models.

#### `/res/img/`

Contains the plots for the training phase of our models.

![[best](https://github.com/dtsbourg/ML_Projects/project2/project_recommender_system/res/img/Deep_Full_Final_1059256_train_117696_test_128_features_Adam_categorical_crossentropy_categorical_.png)]

Figure : Test and Train losses for our best model

#### `/res/model/`

The checkpoint files for the pre-trained models, which can be loaded later on.
The best model is named `best` : it is loaded by default in the prediction pipeline.

#### `/res/pred/`

The prediction provided by our model. The best prediction is named `submission_best.csv`.

### `/notebook/`

A couple notebooks we used for original exploration of the data, methods, and baselines.

### `/data/`

This contains the data used for this project.

#### `/data/data.zip`

This is the training data that was provided to us.

#### `/data/embeddings/embeddings.zip`

This contains the pre-computed embeddings we used in our model.

## Report



> TODO : Document the API
