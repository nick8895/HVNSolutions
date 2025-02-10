# Robot programming HKA WS 2024
Dieses Repository ist im Zuge der Vorlesung Roboterprogrammierung WS24/25 entstanden. Es beinhaltet die genutzte PyBullet-Umgebung sowie die bereitgestellten Tensorflow-Dateien. Für eine unabhängige Arbeitsumgebung, wurden die beiden bereitgestellten Dockerimages zu einem zusammengefasst, sodass sowohl tensorflow als auch PyBulle in einem Container verwendet werden können.  
## Aufgabe 
Die Aufgabe besteht darin, eine beliebige Klotzkonfiguration aus mindestens 5 Klötzen mit einem Roboter abzubauen. Dabei dürfen die aufgebauten Klötze nicht zusammenstürzen.
Zu Auswahl stehen hier Reinforcementlearning oder Immitationlearning. Hierbei ist die Entscheidung auf das Immitationlearning gefallen, bei dem ein Transporter Network genutzt wird. Das Netz lernt aus vorgeführten Greifpositionen, welche in Form von RGB und Tiefendaten aufgenommen werden. 


# Bauen und Starten des Containers 
Aus den zur Verfügung gestellten Dockerfiles wurde ein Dockerfile generiert, damit alle Funktionen aus PyBullet und Tensorflow und damit alle Dateien genutzt werden können. 
## Bau des Containers: 
```bash
./build_image_HVN.sh
```

## Container starten 
```bash
./run_container_HVN.sh
```

Nachdem die Befehle im Terminal eingegeben worden sind, befindet man sich im Workspace des Containers. Dieser ist in je einen Ordner für die PyBullet-Umgebung und die Tensorflow-Anwendungen unterteilt.

## Starten der Simulation 
Zunächst muss man in den entsprechenden Ordner navigieren:
```bash
cd scripts_bullet 
```
Hierin befindet sich die generate_tn_data.py, welche sich mit folgendem Befehl starten lässt: 
```bash
python generate_tn_data.py 
```

Nun wird zufällig eine der drei Klotzkonfigurationen an einer zufälligen Position mit einer zufälligen Rotation generiert im definierten Arbeitsraum gespawnt. Die Anzahl wie oft die einzelnen Konfigurationen abgebaut werden entspricht der in der Config-Datei angegeben Szenen. 

<div align="center">
  <img src="block_variants.png" alt="Bild 1" width="800">
  <p style="font-size:12px; color:gray;"><em>Blockkonfigurationen zur Datengenerierung</em></p>
</div>


# Projektumsetzung 
Im folgenden werden die einzelnen Skripte und deren Aufgaben(?) beschrieben. Die Grundlage übernimmt die "gernerate_tn_data.py" in der Daten generiert und je nach gesetzter Flag im jeweiligen Verzeichnis abgespeichert werden. Die generierten Daten werden durch die "convert_dataset.py" in das richtige Formaet(?) konvertiert. 
generate_tn_data beschreiben, aufbau und beschreibung der einzelnen funktionen/zweck

train_tn_data.py: trainieren des netzes aus RGB und Tiefendaten, welche zuvor als hdf5 datei gespeichert und konvertiert worden sind 

tn_test.py: 3 bilder aufnehmen, orthpgrafische projektion erzeugen -> combined images, modell laden (combined image übergebne), output des netzes, 
## 
