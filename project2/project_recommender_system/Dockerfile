FROM gw000/keras:2.1.1-py3-th-cpu

# install dependencies from debian packages
RUN apt-get update -qq && \
	  apt-get install -y python3-tk

# install dependencies from python packages
RUN pip3 --no-cache-dir install \
    pandas \
    scikit-learn==0.19.1 \
    scipy \
	keras==2.1.2 \
	numpy \
	pandas \
	matplotlib \
	h5py \
	scikit-surprise \
	theano==1.0.0

COPY ./ /cs-433-project/

WORKDIR /cs-433-project
