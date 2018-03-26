DK=docker
IMG=proj2

build:
	$(DK) build . -t $(IMG)

run:
	$(DK) run -it --rm \
	--mount src=$(shell pwd),target=/code,type=bind \
	$(IMG)

ps:
	$(DK) ps