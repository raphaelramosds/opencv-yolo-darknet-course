OIDV4_TOOLKIT=shared/OIDv4_ToolKit
DARKNET = shared/darknet
YOLOV3WEIGHTS = shared/yolov3.weights

.PHONY: build install clean setup oidv4

$(OIDV4_TOOLKIT):
	git clone https://github.com/EscVM/OIDv4_ToolKit.git $(OIDV4_TOOLKIT)

# Etapa 1 - Download do Darknet
$(DARKNET):
	git clone https://github.com/AlexeyAB/darknet.git $(DARKNET)
	find $(DARKNET) -type f -print0 | xargs -0 dos2unix

# Etapa 3 - Baixando os pesos do modelo pré-treinado
$(YOLOV3WEIGHTS):
	wget -O $(YOLOV3WEIGHTS) https://data.pjreddie.com/files/yolov3.weights

# Etapa 2 - Compilando a biblioteca
cpu: $(YOLOV3WEIGHTS) $(DARKNET)
	cd $(DARKNET) && $(MAKE)

# FIXME: getting "CUDA Error: no CUDA-capable device is detected" on jupyter notebook cell
# gpu: $(YOLOV3WEIGHTS) $(DARKNET)
# 	cd $(DARKNET) && \
# 	sed -i 's/^GPU=.*/GPU=1/' Makefile && \
# 	sed -i '/-gencode/d' Makefile && \
# 	sed -i '9iARCH= -gencode arch=compute_50,code=[sm_50,compute_50] -gencode arch=compute_52,code=[sm_52,compute_52] -gencode arch=compute_86,code=sm_86' Makefile && \
# 	$(MAKE)

install:
	poetry install

setup: cpu install

# NOTE. Crie um ambiente python antes de instalar as dependencies do requirements.txt desse repo
# mkdir -p .local/virtualenvs
# python -m venv ~/.local/virtualenvs/oidv4
# source ~/.local/virtualenvs/oidv4/bin/activate
oidv4: $(OIDV4_TOOLKIT)
	cd $(OIDV4_TOOLKIT)

# setup-gpu: gpu install

clean:
	rm -rf $(DARKNET)
	rm -f $(YOLOV3WEIGHTS)