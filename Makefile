IMAGE_NAME=pc-darts
NOW=`date +"%Y%m%d%I%M%S"`
DATA_DIR=/raid/aisin/nas/data

init:
	pre-commit install --hook-type pre-commit --hook-type pre-push

build:
	docker build -t ${IMAGE_NAME} .

start:
	docker run --rm -it --name ${IMAGE_NAME}-${USER}-${NOW} \
						-v ${PWD}:/opt/pc-darts \
						-v ${DATA_DIR}:/opt/data \
						--gpus all \
						${IMAGE_NAME} sh -c "cd /opt/pc-darts && bash"

lint: # Lint all files in this repository
	pre-commit run --all-files --show-diff-on-failure
