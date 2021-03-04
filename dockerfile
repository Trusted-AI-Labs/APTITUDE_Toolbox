# set base image (host OS)
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
&& apt-get install software-properties-common -y \ 
&& add-apt-repository ppa:deadsnakes/ppa -y \
&& apt-get update \
\
&& apt-get install build-essential -y \
&& apt-get install -y --no-install-recommends python3.7-dev python3-pip \
&& update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 \
&& python -m pip install --upgrade pip \
\
&& apt-get install python3.7-tk -y \
&& apt-get install libgtk2.0-0 -y \
&& apt-get install ffmpeg libsm6 libxext6 -y \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements/requirements.txt .
COPY setup.py .
COPY requirements/opencv_contrib_python-4.5.1.48-cp37-cp37m-linux_x86_64.whl .

# install dependencies
RUN pip install --upgrade pip \ 
&& pip install setuptools~=50.3.2 \
&& pip install -r requirements.txt \
&& pip install opencv_contrib_python-4.5.1.48-cp37-cp37m-linux_x86_64.whl \
&& pip install -e .

# copy the content of the local src directory to the working directory
COPY pytb/ pytb/
COPY clients/ clients/
COPY configs/ configs/ 
COPY videos/ videos/
COPY models/ models/

# command to run on container start
CMD python ./clients/main.py -d configs/detect-DM-docker.json -t configs/track-deepsort-docker.json -c configs/classes.json -p videos/cam_10.mp4