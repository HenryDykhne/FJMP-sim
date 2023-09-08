# build docker image
docker build -t fjmp .
# run docker container
docker run --rm -it -u root -v /home/ehdykhne:/ehdykhne --gpus=all --name fjmp_docker_1 l6rowe/luke_apollo