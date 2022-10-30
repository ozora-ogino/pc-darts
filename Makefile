IMAGE_NAME=pc-darts
NOW=`date +"%Y%m%d%I%M%S"`

build:
	docker build -t ${IMAGE_NAME} .

start:
	docker run --rm -it --name ${IMAGE_NAME}-${USER}-${NOW} \
						-v ${PWD}:/opt/pc-darts \
						--gpus all \
						${IMAGE_NAME} sh -c "cd /opt/pc-darts && bash"
