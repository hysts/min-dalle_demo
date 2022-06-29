# min(DALL·E) demo
This is an unofficial demo app for [min(DALL·E)](https://github.com/kuprel/min-dalle).

![](assets/screenshot_01.jpg)
![](assets/screenshot_02.jpg)
![](assets/screenshot_03.jpg)

## Installation
```bash
git clone --recursive https://github.com/hysts/min-dalle_demo
cd min-dalle_demo
docker compose build

cd min-dalle
patch -p1 < ../patch
```

## Download pretrained models
```bash
sudo apt install git-lfs
bash download_models.sh
```

## Run
```baash
docker compose run --rm app
```
