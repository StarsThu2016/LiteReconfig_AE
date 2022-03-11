### Setting Up the Docker on TX2 
Given that TX2 comes with relatively small flash storage, we suggest to install an sdcard (supposing the mount point is $PATH_SDCARD) as an additonal storage space and place the docker working directory on the sdcard with the following command,

```
mkdir $PATH_SDCARD/docker
sudo vi /etc/docker/daemon.json
```
Appending the following content to ```/etc/docker/daemon.json``` so that the docker uses a customized directory to store the containers and images.
```
# The "graph" is the customized docker directory.
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "graph": "$PATH_SDCARD/docker/"
}
```

Confirm that the docker directory has been changed with the following command,
```
sudo service docker restart
sudo ls $PATH_SDCARD/docker  # Some basic docker directories will be setup in the new directory
```


### Build Docker Image for LiteReconfig
The following guide only works for Jetson TX2 with JetPack r32.3.1. To build the docker image, run the following command in which $username is the Docker username and $tag is the name of the image of your choice. This step takes roughly 2.5 hours.

```
# Run the DVFS settings if you haven't already.
sudo nvpmodel -m 0
sudo jetson_clocks
# From docker/
sudo docker build -t $username/$tag .
```

Run the Docker image, where $name should be the name of the container of your choice, hostname should be tx2-2, $PATH_TO_ILSVRC2015_OUTSIDE_DOCKER should be the path to the ILSVRC 2015 dataset on the device.
```
sudo docker run --gpus all --name $name -it --hostname tx2-2 -v $PATH_TO_ILSVRC2015_OUTSIDE_DOCKER:/ILSVRC2015 $username/$tag
```

Run the following commands inside a docker container. They are the additional setup that are not in the dockerfile.
```
(base) root@tx2-2:/LiteReconfig_AE# conda activate ae
(ae) root@tx2-2:/LiteReconfig_AE# cd ../torchvision
(ae) root@tx2-2:/LiteReconfig_AE# export BUILD_VERSION=0.5.0
(ae) root@tx2-2:/LiteReconfig_AE# python setup.py install
(ae) root@tx2-2:/LiteReconfig_AE# mv /models ./
```

Run LiteReconfig with the following command,
```
(ae) root@tx2-2:/LiteReconfig_AE# python LiteReconfig.py --gl 0 \
  --lat_req 33.3 --mobile_device=tx2 \
  --output=test/executor_LiteReconfig.txt \
  --dataset_prefix=/ILSVRC2015
```
