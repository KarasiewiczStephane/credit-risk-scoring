.PHONY: install test lint clean run docker-build docker-run docker-test docker-compose-up docker-compose-down

IMAGE_NAME := credit-risk-scoring
IMAGE_TAG := latest

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.main

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	docker run -p 8000:8000 --rm $(IMAGE_NAME):$(IMAGE_TAG)

docker-test:
	@echo "Testing container health..."
	docker run -d --name test-container -p 8000:8000 $(IMAGE_NAME):$(IMAGE_TAG)
	sleep 5
	curl -f http://localhost:8000/health || (docker logs test-container && exit 1)
	docker stop test-container
	docker rm test-container

docker-compose-up:
	docker compose up --build

docker-compose-down:
	docker compose down
