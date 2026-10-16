# Shortcuts. On Windows without make, copy the commands into your terminal.
export PYTHONPATH := src:.

install:            ## install all dependencies
	pip install -r requirements.txt

data:               ## download + clean the dataset
	python -m heart.data

eda:                ## make EDA charts in reports/figures
	python -m heart.eda

train:              ## train, tune and track models with MLflow
	python -m heart.train

mlflow-ui:          ## open the MLflow UI at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

lint:
	flake8 src api tests

test:
	python -m pytest -v --cov=src/heart --cov=api

serve:              ## run the API without Docker
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t heart-disease-api:1.0.0 .

docker-run:
	docker run --rm -p 8000:8000 --name heart-api heart-disease-api:1.0.0

stack:              ## API + Prometheus + Grafana
	docker compose up --build -d

k8s-deploy:
	kubectl apply -f k8s/00-namespace.yaml -f k8s/01-deployment.yaml -f k8s/02-service.yaml

all: data eda train lint test
