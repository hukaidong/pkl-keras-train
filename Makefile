DOCKER_VERBOSE_ARGS :=
ifneq ($(VERBOSE),)
	DOCKER_VERBOSE_ARGS := --progress=plain
endif

build-sifs: keras.Dockerfile | sifs
	docker build $(DOCKER_VERBOSE_ARGS) -f $< -t keras:pkl-keras .
	sudo singularity build sifs/keras.sif docker-daemon://keras:pkl-keras

jupyter-notebook:
	singularity run --nv -B/media/kaidong:/media/kaidong:ro sifs/keras.sif jupyter-lab > jupyter.log 2>&1 &

sifs:
	mkdir -p sifs
