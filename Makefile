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
build: $(WEIGHTS) $(DARKNET_DIR)
	cd $(DARKNET_DIR) && $(MAKE)

install:
	poetry install

setup: build install

clean:
	rm -rf $(DARKNET_DIR)
	rm -f $(WEIGHTS)