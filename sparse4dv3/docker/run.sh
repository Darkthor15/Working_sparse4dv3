#!/bin/bash
# set X
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

IMAGE="sparse4dv3_nuscenes:ros"
#IMAGE="sparse4dv3_nuscenes:ros_kai3"
IS_EXEC=$(docker ps | grep $IMAGE)
if [ ${#IS_EXEC} -ne 0 ]; then
    docker exec -it sparse4dv3_nuscenes bash
else
    # run docker container
    docker run -it \
            --runtime nvidia \
            --rm \
            --privileged \
            --net=host \
            --ipc=host \
            --pid=host \
            --env="no_proxy=$no_proxy" \
            --env="http_proxy=$http_proxy" \
            --env="https_proxy=$https_proxy" \
            --volume=$XSOCK:$XSOCK:rw \
            --volume=$XAUTH:$XAUTH:rw \
            --volume=/dev:/dev:rw \
            --volume="$(realpath ..):/home/workspace/sparse4dv3:rw" \
            --volume="/home/$USER/.ros/:/home/root/.ros/:rw" \
            --workdir="/home/workspace/sparse4dv3" \
            --env="XAUTHORITY=${XAUTH}" \
            --env="DISPLAY=${DISPLAY}" \
            --env=TERM=xterm-256color \
            --env=QT_X11_NO_MITSHM=1 \
            --shm-size 32G \
            --name sparse4dv3_nuscenes \
            $IMAGE \
            bash --login
fi