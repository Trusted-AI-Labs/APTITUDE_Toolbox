# set base image (host OS)
FROM python:3.7

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .
COPY setup.py .

# install dependencies
RUN pip install -r requirements.txt \
&& pip install -e .

RUN apt-get update \
&& apt-get install ffmpeg libsm6 libxext6 -y

# copy the content of the local src directory to the working directory
COPY pytb/ pytb/
COPY clients/ clients/
COPY configs/ configs/ 
COPY videos/ videos/
COPY models/ models/

# command to run on container start
CMD python ./clients/main.py -d configs/detect-DM-docker.json -t configs/track-deepsort-docker.json -c configs/classes.json -p videos/cam_10.mp4