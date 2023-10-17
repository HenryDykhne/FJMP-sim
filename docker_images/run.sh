# build docker image
docker build -t fjmp-sim-new .
# run docker container
docker run --rm -it -u root -v /home/ehdykhne:/ehdykhne --gpus=all --name fjmp_docker_2 fjmp-sim-new