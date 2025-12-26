OIDV4_TOOLKIT=shared/OIDv4_ToolKit
DARKNET = shared/darknet
YOLOV3WEIGHTS = shared/yolov3.weights

.PHONY: install clean setup-darknet oidv4

$(OIDV4_TOOLKIT):
	git clone https://github.com/EscVM/OIDv4_ToolKit.git $(OIDV4_TOOLKIT)

# Download do Darknet
$(DARKNET):
	git clone https://github.com/AlexeyAB/darknet.git $(DARKNET)
	find $(DARKNET) -type f -print0 | xargs -0 dos2unix

# Download dos pesos do modelo pré-treinado
$(YOLOV3WEIGHTS):
	wget -O $(YOLOV3WEIGHTS) https://data.pjreddie.com/files/yolov3.weights

# Compilacao da biblioteca
darknet-cpu: $(YOLOV3WEIGHTS) $(DARKNET)
	cd $(DARKNET) && $(MAKE)

darknet-gpu: $(YOLOV3WEIGHTS) $(DARKNET)
	cd $(DARKNET) && \
	sed -i 's/^GPU=.*/GPU=1/' Makefile && \
	sed -i 's/^OPENCV=.*/OPENCV=1/' Makefile && \
	sed -i 's/^CUDNN=.*/CUDNN=1/' Makefile && \
	$(MAKE)

# NOTE. Crie um ambiente python antes de instalar as dependencies do requirements.txt desse repo
# mkdir -p .local/virtualenvs
# python -m venv ~/.local/virtualenvs/oidv4
# source ~/.local/virtualenvs/oidv4/bin/activate
oidv4: $(OIDV4_TOOLKIT)
	cp src/scripts/converter_annotations.py \
	   src/scripts/gerar_test.py \
	   src/scripts/gerar_train.py \
	   $(OIDV4_TOOLKIT)


clean:
	rm -rf $(DARKNET)
	rm -f $(YOLOV3WEIGHTS)