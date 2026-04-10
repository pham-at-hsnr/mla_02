# Duckietown Learning Experience (LX)

Dies ist eine angepasste Duckietown Learning Experience zum Thema Regression.

Im Mittelpunkt steht eine konkrete Frage aus der Robotik:
Wie kann ein Roboter aus Bildinformationen abschätzen, wie weit ein erkanntes Objekt entfernt ist,
und diese Information anschliessend für sein Verhalten nutzen?

## Ziel der Learning Experience

Das übergeordnete Ziel ist es, ein Distanzschätzungsmodell von Grund auf zu entwickeln,
zu trainieren und in einen virtüllen Duckiebot einzubetten.

Die konkrete Anwendung ist eine Distanzschätzung für detektierte Objekte.
Die (sehr einfache) Objektdetektion selbst wird in dieser Learning Experience als Blackbox vorgegeben.
Sie müssen also keinen eigenen Detektor trainieren oder verbessern.
Stattdessen liegt der Fokus darauf, aus den Merkmalen einer vorhandenen Detektion,
insbesondere der Breite und Höhe einer Bounding Box im Kamerabild, die Entfernung zum Objekt abzuschätzen.

## Learning Outcomes

Nach Bearbeitung dieser Learning Experience sollten Sie in der Lage sein:

- den Unterschied zwischen Klassifikation, Objekterkennung und Regression zu erklären
- zu argumentieren, inwiefern Bounding-Box-Merkmale für eine einfache Distanzschätzung nutzbar sind und welche Limitierungen es gibt
- einen Trainingsprozess mit Gradient Descent konzeptionell nachzuvollziehen und praktisch umzusetzen
- Gewichte eines gelernten Regressionsmodells mit NumPy zu exportieren und für die Inferenz wieder zu laden

## Ablauf und Struktur

Die Anleitungen und Aufgaben befinden sich hauptsächlich in den Jupyter-Notebooks in diesem Repository. Die Notebooks baün aufeinander auf und führen Schritt für Schritt bis zur Integration.

Die Struktur ist in zwei thematische Blöcke gegliedert:

### 1. Training

Im Notebook `notebooks/01-Training/training.ipynb` trainieren Sie das Distanzmodell. Die Objekterkennung ist bereits gegeben und liefert Bounding Boxes. Die Aufgabe besteht darin, aus Breite und Höhe dieser Bounding Boxes ein Regressionsmodell zu lernen, das die Distanz zum Objekt abschätzt. Dabei soll das Modell explizit mit Gradient Descent trainiert werden.

Ein wichtiger Bestandteil dieses Schritts ist ausserdem der Export der gelernten Gewichte mit NumPy, damit sie später ausserhalb des Notebooks für die Inferenz verwendet werden können.

### 2. Integration

Im Notebook `notebooks/02-Integration/integration.ipynb` wird beschrieben, wie das trainierte Modell in den Duckiebot eingebunden wird.
Ziel ist es, die Distanzschätzung im laufenden System nutzbar zu machen,
während die Objekterkennung weiterhin als bereits vorhandene Komponente behandelt wird.

## Export und Import der Gewichte mit NumPy

Für diese Learning Experience sollen die Gewichte bewusst einfach und transparent gespeichert werden. Verwenden Sie dafür NumPy statt eines komplexeren Modellformats.

Ein möglicher Export im Training-Notebook sieht so aus:

```python
np.savez("numpy_weights.npz", weights=weights, bias=bias)
```

Dabei gilt:

- `weights` enthält zum Beispiel die gelernten Regressionsgewichte für Breite und Höhe
- `bias` enthält den Bias-Term des Modells
- die Datei `numpy_weights.npz` kann später direkt in der Codebasis geladen werden

Der Import in `student_distance_estimator.py` erfolgt dann mit `np.load(...)`, zum Beispiel:

```python
params = np.load(DISTANCE_MODEL_PATH)
weights = params["weights"]
bias = params["bias"]
```

## Credits

Dieses Repository basiert auf der Duckietown Object Detection Learning Experience, wurde jedoch für Lehrzwecke geklont und inhaltlich angepasst.

Zur Transparenz hier das ursprüngliche Duckietown-Repository:

https://github.com/duckietown/lx-object-detection
