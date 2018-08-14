PIP :=pip
PYTHON :=python
NAME :=robust_offline_contextual_bandits
LIB_NAME :=robust_offline_contextual_bandits

.PHONY: default
default: install

.PHONY: test
test: setup.py
	$(PYTHON) setup.py -q test --addopts '-x $(ARGS)'

.PHONY: test-cov
test-cov: ARGS=--cov $(LIB_NAME) --cov-report term:skip-covered
test-cov: test

.PHONY: testv
testv: ARGS=-vs
testv: test

.PHONY: install
install: requirements.txt
	$(PIP) install -r requirements.txt $(ARGS)

.PHONY: install-dev
install-dev: requirements.txt
	$(PIP) install -r requirements.txt -e . $(ARGS)

.PHONY: clean
clean:
	-find . -name "*.pyc" -delete
	-find . -name "__pycache__" -delete
	-find $(LIB_NAME) -name "*.so" -delete
	-rm -rf .cache *.egg .eggs *.egg-info build 2> /dev/null

.PHONY: clean-tmp
clean-tmp:
	-rm -rf tmp

.PHONY: build
build: .build

.build: setup.py $(LIB_NAME)/__init__.py
	$(PYTHON) setup.py sdist bdist_wheel
	touch $@
