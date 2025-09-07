DARKNET_DIR = shared/darknet
WEIGHTS = shared/yolov3.weights

.PHONY: build install clean setup

# Etapa 1 - Download do Darknet
$(DARKNET_DIR):
	git clone https://github.com/pjreddie/darknet.git $(DARKNET_DIR)
	find $(DARKNET_DIR) -type f -print0 | xargs -0 dos2unix

$(WEIGHTS):
	wget -O $(WEIGHTS) https://data.pjreddie.com/files/yolov3.weights

# Etapa 2 - Compilando a biblioteca
# Etapa 3 - Baixando os pesos do modelo pré-treinado
cpu: $(WEIGHTS) $(DARKNET_DIR)
	cd $(DARKNET_DIR) && $(MAKE)

# FIXME: getting "CUDA Error: no CUDA-capable device is detected" on jupyter notebook cell
gpu: $(WEIGHTS) $(DARKNET_DIR)
	cd $(DARKNET_DIR) && \
	sed -i 's/^GPU=.*/GPU=1/' Makefile && \
	sed -i '/-gencode/d' Makefile && \
	sed -i '9iARCH= -gencode arch=compute_50,code=[sm_50,compute_50] -gencode arch=compute_52,code=[sm_52,compute_52] -gencode arch=compute_86,code=sm_86' Makefile && \
	$(MAKE)

install:
	poetry install

setup: cpu install

setup-gpu: gpu install

clean:
	rm -rf $(DARKNET_DIR)
	rm -f $(WEIGHTS)
