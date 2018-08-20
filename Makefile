DOCKER_IMAGE_VERSION=$$(cat VERSION)
DOCKER_IMAGE_NAME=knjcode/imgdupes
DOCKER_IMAGE_TAGNAME=$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)

default: build

build:
	python setup.py sdist
	docker pull python:3.6.6-slim-stretch
	docker build --build-arg VERSION=$(DOCKER_IMAGE_VERSION) -t $(DOCKER_IMAGE_TAGNAME) .
	docker tag $(DOCKER_IMAGE_TAGNAME) $(DOCKER_IMAGE_NAME):latest

rebuild:
	python setup.py sdist
	docker pull python:3.6.6-slim-stretch
	docker build --build-arg VERSION=$(DOCKER_IMAGE_VERSION) --no-cache -t $(DOCKER_IMAGE_TAGNAME) .
	docker tag $(DOCKER_IMAGE_TAGNAME) $(DOCKER_IMAGE_NAME):latest

push:
	docker push $(DOCKER_IMAGE_TAGNAME)
	docker push $(DOCKER_IMAGE_NAME):latest

test:
	docker-compose run imgdupes -h
