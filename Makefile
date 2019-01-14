ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON = python3.7

install: install_backend install_frontend

install_backend: install_virtualenv setup_database

install_virtualenv:
	virtualenv $(ROOT_DIR)/env; \
	source $(ROOT_DIR)/env/bin/activate; \
	pip3 install -r $(ROOT_DIR)/requirements.txt;

setup_database:
	source $(ROOT_DIR)/env/bin/activate; \
	cd $(ROOT_DIR)/backend/src; \
	$(PYTHON) manage.py makemigrations; \
	$(PYTHON) manage.py migrate; \
	$(PYTHON) manage.py loaddata db.json;

install_frontend:
	cd $(ROOT_DIR)/frontend/gui; \
	npm install --no-optional

run_backend:
	source $(ROOT_DIR)/env/bin/activate; \
	cd $(ROOT_DIR)/backend/src; \
	$(PYTHON) manage.py runserver

run_frontend:
	cd $(ROOT_DIR)/frontend/gui; \
	npm start
