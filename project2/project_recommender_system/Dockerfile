FROM gw000/keras

# install dependencies from debian packages
RUN apt-get update -qq && \
	  apt-get install -y python3-tk python3-pip

RUN python3 -m pip install --upgrade ez_setup \
		setuptools

# install dependencies from python packages
RUN python3 -m pip --no-cache-dir install --upgrade \
    pandas \
    scikit-learn==0.19.1 \
    scipy \
		keras==2.1.2 \
		numpy>=1.11.2 \
		cython \
		pandas \
		matplotlib \
		h5py \
		theano==1.0.0\
		tensorflow

RUN pip3 --no-cache-dir install scikit-surprise

COPY ./ /cs-433-project/

WORKDIR /cs-433-project
