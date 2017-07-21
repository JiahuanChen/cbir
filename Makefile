.PHONY: build

all: build

build:
	docker build -t cbir .

run-interactive:
	docker run \
		-v "${PWD}:/app" \
		-it \
		--rm \
		cbir \
		bash