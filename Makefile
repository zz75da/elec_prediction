.PHONY: help install extract-data train predict test test-unit test-integration quality-gate up down build logs dvc-pull dvc-push clean

help:
	@echo "elec_prediction — targets:"
	@echo "  install          Install root requirements.txt in the current venv"
	@echo "  extract-data     Re-run the raw-data ETL (scripts/extract_raw_data.py)"
	@echo "  train            Call POST /train on train-api (docker compose must be up)"
	@echo "  predict          Call POST /predict on predict-api (horizon=12)"
	@echo "  test             Run the full pytest suite (unit + integration)"
	@echo "  test-unit        Run unit tests only"
	@echo "  quality-gate     Run the model quality gate (train-api/tests/test_model_quality.py)"
	@echo "  up               docker compose up -d (train-api, predict-api, streamlit, prometheus, grafana)"
	@echo "  down             docker compose down"
	@echo "  build            docker compose build"
	@echo "  logs             docker compose logs -f"
	@echo "  dvc-pull         dvc pull (data/artifacts from DagsHub remote)"
	@echo "  dvc-push         dvc push"

install:
	pip install -r requirements.txt

extract-data:
	python scripts/extract_raw_data.py

train:
	curl -s -X POST http://localhost:5010/train -H "Content-Type: application/json" -d '{}' | python -m json.tool

predict:
	curl -s -X POST http://localhost:5011/predict -H "Content-Type: application/json" -d '{"horizon": 12}' | python -m json.tool

test:
	pytest tests/ train-api/tests/ -v

test-unit:
	pytest tests/unit/ train-api/tests/ -v

test-integration:
	pytest tests/integration/ -v -m integration

quality-gate:
	pytest train-api/tests/test_model_quality.py -v

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

dvc-pull:
	dvc pull

dvc-push:
	dvc push

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
