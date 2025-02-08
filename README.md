# Robot programming HKA WS 2024
Dieses Repository ist im Zuge der Vorlesung Roboterprogrammierung WS24/25 entstanden. Es beinhaltet die genutzte PyBullet-Umgebung sowie die bereitgestellten Tensorflow-Dateien. Für eine unabhängige Arbeitsumgebung, wurden die beiden bereitgestellten Dockerimages zu einem zusammengefasst, sodass sowohl tensorflow als auch PyBulle in einem Container verwendet werden können.  
## Aufgabe 
Die Aufgabe besteht darin, eine beliebige Klotzkonfiguration aus mindestens 5 Klötzen mit einem Roboter abzubauen. Dabei dürfen die aufgebauten Klötze nicht zusammenstürzen.
Zu Auswahl stehen hier Reinforcementlearning oder Immitationlearning. Hierbei ist die Entscheidung auf das Immitationlearning gefallen, bei dem ein Transporter Network genutzt wird. Das Netz lernt aus vorgeführten Greifpositionen, welche in Form von RGB und Tiefendaten aufgenommen werden. 

## build des Containers
Aus den zur Verfügung gestellten Dockerfiles wurde ein Dockerfile generiert, damit alle Funktionen aus PyBullet und Tensorflow und damit alle Dateien genutzt werden können. 
Bau des Containers: 
**Requirements:** have docker installed i

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
