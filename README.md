# Duckietown Learning Experience

Dies ist eine angepasste Duckietown Learning Experience zum Thema maschinelles Lernen fuer robotische Wahrnehmung und einfache handlungsrelevante Entscheidungslogik.

Im Mittelpunkt steht eine konkrete Frage aus der Robotik: Wie kann ein Roboter aus Bildinformationen abschaetzen, wie weit ein erkanntes Objekt entfernt ist, und diese Information anschliessend fuer sein Verhalten nutzen?

Das Repository verbindet dafuer drei Ebenen:

- eine vorgegebene Objekterkennung als Blackbox
- das Training eines einfachen Regressionsmodells mit Gradient Descent
- die Integration des gelernten Distanzmodells in einen Duckietown-Agenten

## Ziel der Learning Experience

Das uebergeordnete Ziel ist es, ein Distanzschaetzungsmodell von Grund auf zu entwickeln, zu trainieren und in eine lauffaehige Agentenlogik einzubetten.

Die konkrete Anwendung ist eine Distanzschaetzung fuer erkannte Objekte. Die Objekterkennung selbst wird in dieser Learning Experience als Blackbox vorgegeben. Studierende muessen also keinen eigenen Detektor trainieren oder verbessern. Stattdessen liegt der Fokus darauf, aus den Merkmalen einer vorhandenen Detektion, insbesondere der Breite und Hoehe einer Bounding Box im Kamerabild, die Entfernung zum Objekt abzuschaetzen.

Diese Schaetzung wird spaeter genutzt, um auf Hindernisse oder Duckies angemessen zu reagieren. Die Learning Experience soll damit nicht nur zeigen, wie ein Regressionsmodell trainiert wird, sondern auch, wie ein ML-Baustein Teil eines groesseren robotischen Systems wird.

## Learning Outcomes

Nach Bearbeitung dieser Learning Experience solltest du in der Lage sein:

- den Unterschied zwischen Klassifikation, Objekterkennung und Regression in einem robotischen Kontext zu erklaeren
- zu erklaeren, warum die Objekterkennung in dieser Aufgabe bewusst als gegebene Blackbox behandelt wird
- zu begruenden, warum Bounding-Box-Merkmale fuer eine einfache Distanzschaetzung nutzbar sind
- einen Trainingsprozess mit Gradient Descent konzeptionell nachzuvollziehen und praktisch umzusetzen
- Gewichte eines gelernten Regressionsmodells mit NumPy zu exportieren und fuer Inferenz wieder zu laden
- eine Distanzfunktion in bestehende Agentenlogik zu integrieren
- die Rolle von Konfidenzschwellen, Stop-Distanzen und Kamerageometrie fuer sicheres Verhalten zu verstehen
- die Grenzen eines einfachen Modells zu reflektieren, zum Beispiel Sensitivitaet gegen Perspektive, Kameraposition oder schlechte Detektionen

## Didaktischer Fokus

Diese Einheit ist bewusst anwendungsnah aufgebaut. Statt ein abstraktes Regressionsproblem isoliert zu betrachten, arbeitest du an einer Aufgabe, die direkt in einem robotischen Wahrnehmungs- und Kontrollsystem verwendet wird.

Dadurch lernst du nicht nur mathematische Grundlagen, sondern auch den Transfer in ein Software-System mit mehreren Komponenten:

- Wahrnehmung ueber Kamerabilder
- Objekterkennung ueber eine vorgegebene Blackbox
- Distanzschaetzung ueber ein selbst trainiertes Regressionsmodell
- Verhaltensentscheidung des Agenten, zum Beispiel Stoppen bei zu geringem Abstand

## Ablauf und Struktur

Die Anleitungen und Aufgaben befinden sich hauptsaechlich in den Jupyter-Notebooks in diesem Repository. Die Notebooks bauen fachlich aufeinander auf und fuehren Schritt fuer Schritt von den Grundlagen bis zur Integration.

Die Struktur ist in zwei thematische Bloecke gegliedert:

### 1. Training

Im Notebook `notebooks/01-Training/training.ipynb` trainierst du das Distanzmodell. Die Objekterkennung ist bereits gegeben und liefert Bounding Boxes. Deine Aufgabe besteht darin, aus Breite und Hoehe dieser Bounding Boxes ein Regressionsmodell zu lernen, das die Distanz zum Objekt abschaetzt. Dabei soll das Modell explizit mit Gradient Descent trainiert werden.

Ein wichtiger Bestandteil dieses Schritts ist ausserdem der Export der gelernten Gewichte mit NumPy, damit sie spaeter ausserhalb des Notebooks fuer die Inferenz verwendet werden koennen.

### 2. Integration

Im Notebook `notebooks/02-Integration/integration.ipynb` wird das trainierte Modell in den Agenten eingebunden. Ziel ist es, die Distanzschaetzung im laufenden System nutzbar zu machen, waehrend die Objekterkennung weiterhin als bereits vorhandene Komponente behandelt wird.

## Verbindung zum Code

Neben den Notebooks enthaelt das Repository Python-Code fuer die spaetere Systemintegration.

Besonders relevant sind:

- `packages/solution/student_distance_estimator.py`
  Hier implementierst oder vervollstaendigst du die Distanzschaetzung auf Basis der trainierten Modellgewichte. Diese Gewichte sollen zuvor mit NumPy aus dem Trainings-Notebook exportiert und hier wieder geladen werden.
- `packages/solution/model.py`
  Diese Datei verbindet die vorgegebene Objekterkennung, die Distanzlogik und das Agentenverhalten.
- `packages/solution/config.py`
  Hier sind wichtige Parameter wie Modellpfade, Konfidenzschwellen und Stop-Distanzen definiert.

Damit wird sichtbar, wie aus Notebook-Ergebnissen ein wiederverwendbarer Bestandteil eines Robotik-Stacks entsteht.

## Empfohlener Workflow

Eine sinnvolle Bearbeitungsreihenfolge ist:

1. Verstehe die Aufgabenaufteilung: Detektion ist gegeben, Distanzschaetzung ist dein eigentlicher ML-Teil.
2. Trainiere im Notebook ein Regressionsmodell nur auf Basis von `pixel_width` und `pixel_height`.
3. Speichere die gelernten Gewichte mit NumPy, zum Beispiel in einer `.npz`-Datei.
4. Lege diese Datei an der erwarteten Stelle im Projekt ab.
5. Lade die Gewichte in `student_distance_estimator.py` mit `numpy.load(...)`.
6. Verwende die geladenen Gewichte in der Inferenzfunktion `estimate_distance(...)`.
7. Teste die Integration im Agenten und beobachte das Verhalten kritisch.

## Export und Import der Gewichte mit NumPy

Fuer diese Learning Experience sollen die Gewichte bewusst einfach und transparent gespeichert werden. Verwende dafuer NumPy statt eines komplexeren Modellformats.

Ein moeglicher Export im Training-Notebook sieht so aus:

```python
np.savez("numpy_weights.npz", weights=weights, bias=bias)
```

Dabei gilt:

- `weights` enthaelt zum Beispiel die gelernten Regressionsgewichte fuer Breite und Hoehe
- `bias` enthaelt den Bias-Term des Modells
- die Datei `numpy_weights.npz` kann spaeter direkt in der Codebasis geladen werden

Der Import in `student_distance_estimator.py` erfolgt dann mit `numpy.load(...)`, zum Beispiel:

```python
params = np.load(DISTANCE_MODEL_PATH)
weights = params["weights"]
bias = params["bias"]
```

Wichtig ist, dass Export und Import dieselben Schluessel verwenden. Wenn du also mit `weights=` und `bias=` speicherst, musst du spaeter auch genau diese Namen beim Laden verwenden.

## Was du fachlich mitnehmen sollst

Diese Learning Experience ist nicht nur ein Coding Exercise. Sie soll dir helfen, ein belastbares Verstaendnis fuer den gesamten ML-Lebenszyklus in einer kleinen, aber realistischen Robotik-Anwendung aufzubauen:

- Problemformulierung
- Merkmalswahl
- Datennutzung
- Modelltraining
- Gewichtsexport und spaetere Wiederverwendung
- Inferenz
- Integration
- Bewertung von Fehlverhalten und Grenzen

Gerade dieser durchgehende Zusammenhang ist fuer spaetere ML-Projekte wichtig: Ein Modell ist erst dann wirklich nuetzlich, wenn klar ist, wie es mit Sensorik, Software-Architektur und Entscheidungslogik zusammenspielt.

## Bearbeitungshinweise

Falls gewuenscht, kannst du die Notebooks auch in Google Colab bearbeiten und ausfuehren. Die Dateien und Assets in diesem Repository sind so organisiert, dass sowohl experimentelles Arbeiten in Notebooks als auch die Einbindung in den Duckietown-Code moeglich sind.

Es lohnt sich, waehrend der Bearbeitung nicht nur auf "funktioniert" oder "funktioniert nicht" zu schauen, sondern systematisch zu reflektieren:

- Welche Eingaben beeinflussen die Distanzschaetzung besonders stark?
- Welche Fehler entstehen durch ungenaue Detektionen?
- Wie muessen Gewichte gespeichert werden, damit Training und Inferenz konsistent zusammenpassen?
- Wie robust ist das Verhalten gegen veraenderte Perspektiven oder Bildbedingungen?
- Welche Vereinfachungen wurden fuer diese Lernumgebung bewusst gemacht?

## Transparenz

Dieses Repository basiert auf der Duckietown Object Detection Learning Experience, wurde jedoch fuer Lehrzwecke geklont und inhaltlich angepasst.

Zur Transparenz siehe das urspruengliche Duckietown-Repository:

https://github.com/duckietown/lx-object-detection

## Credits

Grundlage: Duckietown Object Detection Learning Experience.
