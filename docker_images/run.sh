# build docker image
docker build -t fjmp-sim-0 .
# run docker container
docker run --rm -it -u root -v /home/ehdykhne:/ehdykhne --gpus=all --name fjmp_docker_2 fjmp-sim-0