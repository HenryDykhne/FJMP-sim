# build docker image
docker build -t fjmp-sim-1 .
# run docker container
docker run --rm -it -u root -v /home/ehdykhne:/ehdykhne --gpus=all --name fjmp_docker_3 fjmp-sim-1