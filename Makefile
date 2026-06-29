.PHONY: install dev-install setup pretrain sft rlhf train-all api ui test lint format clean

install:
	pip install -e .

dev-install:
	pip install -e .
	pip install -r requirements-dev.txt

setup:
	python manage.py setup

pretrain:
	python manage.py pretrain

sft:
	python manage.py sft

rlhf:
	python manage.py rlhf

train-all:
	python manage.py setup
	python manage.py pretrain
	python manage.py sft
	python manage.py rlhf

api:
	python manage.py api

ui:
	python manage.py ui

test:
	python manage.py test

lint:
	flake8 src/ tests/

format:
	black src/ tests/

clean:
	python manage.py clean

