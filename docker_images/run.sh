# build docker image
docker build -t fjmp-sim .
# run docker container
docker run --rm -it -u root -v /home/ehdykhne:/ehdykhne --gpus=all --name fjmp_docker_1 fjmp-sim