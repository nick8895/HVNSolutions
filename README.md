# Robot programming HKA WS 2024
Dieses Repository ist im Zuge der Vorlesung Roboterprogrammierung WS24/25 entstanden. Es beinhaltet die genutzte PyBullet-Umgebung sowie die bereitgestellten Tensorflow-Dateien. Für eine unabhängige Arbeitsumgebung, wurden die beiden bereitgestellten Dockerimages zu einem zusammengefasst, sodass sowohl tensorflow als auch PyBulle in einem Container verwendet werden können.  
## Aufgabe 


### Environment setup

**Requirements:** have docker installed including the post-installation steps.

**Note:** The default settings are for nvidia GPU support. If you don't have an nvidia GPU, open up `build_image.sh` and set the `render` argument to `base`. Also, remove the `--gpus all` flag from the `docker run` command in `run_container.sh`.

Build the docker image with

```bash
./build_image.sh
```

Run the container with
```bash
./run_container.sh
```

Check whether you can open a window from the container by running
```bash
python view_noise_image.py
```
A window should pop up showing a random noise image. You can close it by pressing any key, while the window is focused.

### Basics
Check out the `basics.py` script to get familiar with the API we use and the `Affine` class.

If you want to understand a bit more how everything works under the hood, check out the scripts in the `bullet_env` folder as well as the `transform.py` file.
