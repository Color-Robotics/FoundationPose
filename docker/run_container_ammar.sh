CONTAINER_NAME=foundationpose_ammar
docker rm -f ${CONTAINER_NAME}
DIR=$(pwd)/../
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name ${CONTAINER_NAME}  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X12-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE ${CONTAINER_NAME}:latest bash -c "cd $DIR && bash"
