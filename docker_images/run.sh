# build docker image
docker build -t fjmp-sim-2 .
# run docker container
docker run --rm -it -u root -v /home/ehdykhne:/ehdykhne --gpus=all --name henry_fjmp_docker fjmp-sim-2