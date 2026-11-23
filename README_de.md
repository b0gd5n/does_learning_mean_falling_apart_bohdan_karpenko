# Integrat

Code zur Arbeit „Lernen heißt zerfallen? Das Verhalten von φₛ und Φ beim Training mit Gradientenabstieg“ (Bohdan Karpenko).

Ein Integrat ist ein rekurrentes Netz aus stochastischen binären Knoten. Es wird mit Gradientenabstieg trainiert, und seine Integration nach der Theorie der integrierten Information (IIT 4.0) wird mit der Bibliothek PyPhi exakt berechnet. Verglichen werden Ring-Initialisierung und zufällige Initialisierung bei Kernen aus 2, 3 und 4 Knoten, sieben Aufgaben und fünf Verzögerungen, insgesamt 840 Netze.

## Dateien

| Datei | Was sie macht |
|---|---|
| `integrats.py` | Die Klasse `Integrat`, Vorwärts- und Rückwärtsdurchlauf, Optimierer AdamW, Berechnung von φₛ und Φ mit PyPhi |
| `research.py` | Alle Versuche. Atlas untrainierter Kerne, Training von 840 Netzen und Mischen der Gewichte. Schreibt CSV-Dateien in den Ordner `data`, je Aufgabe eigene Dateien |
| `make_plots.py` | Zeichnet die Diagramme aus den CSV-Dateien in `data` und speichert sie auf Englisch und Deutsch im Ordner `plots` |
| `statistics.py` | Berechnet die gesamte Statistik der Arbeit aus den CSV-Dateien in `data`, also die Wilcoxon-Tests und die Korrelationen nach Pearson und Spearman |
| `requirements.txt` | Benötigte Bibliotheken |

## Installation

```
python -m pip install -r requirements.txt
```

PyPhi wird aus dem Zweig `feature/iit-4.0` installiert, festgelegt auf den Commit `b78d0e342`, weil die normale PyPhi-Version keine IIT 4.0 enthält.

## Ausführen

```
python research.py atlas
python research.py and
python research.py or
python research.py xor
python research.py one_back_xor
python research.py replay
python research.py replay_twice
python research.py noise
python make_plots.py
python statistics.py
```

`research.py atlas` berechnet die untrainierten Kerne, jeder andere Befehl trainiert alle Netze einer Aufgabe. Die Aufgaben sind voneinander unabhängig und können gleichzeitig in getrennten Terminals laufen. Das Training dauert etwa einen Tag, weil φₛ und Φ für jeden Zustand des Kerns exakt berechnet werden. Vor einem neuen Durchlauf die alten Dateien aus `data` an einen anderen Ort kopieren, sonst werden sie überschrieben. `statistics.py` gibt alle p-Werte und Korrelationskoeffizienten der Arbeit aus.

## Ergebnisse

Im Ordner `data` liegen

- `atlas.csv` mit φₛ und Φ untrainierter Kerne aus 2, 3, 4 und 5 Knoten
- `training_accuracy_<Aufgabe>.csv` mit der Genauigkeit jedes Netzes vor dem Training, alle 1024 Epochen und nach jeder der 8 Mischungen der Gewichte
- `training_integration_<Aufgabe>.csv` mit φₛ und Φ des ganzen Kerns für jeden Zustand zu denselben Zeitpunkten
- `training_weights_<Aufgabe>.csv` mit den Gewichten jedes Netzes am Ende des Trainings

Im Ordner `plots` liegen die Diagramme in den Unterordnern `en` und `de` sowie die Tabelle `summary.csv` mit den Mittelwerten vor dem Training, nach dem Training und nach dem Mischen.

Die Arbeit liegt im Ordner `article` auf Englisch und Deutsch.

## Reproduzierbarkeit

Alle Zufallszahlen werden durch die Startwerte 0, 1, 2 und 3 des Zufallsgenerators festgelegt. Die Testsequenzen stammen aus einem eigenen Generator mit dem Startwert 12345, daher beeinflusst die Messung der Genauigkeit das Training nicht. Mit denselben Bibliotheksversionen sollten die Ergebnisse mit der Arbeit übereinstimmen.

## Einsatz von KI

Der Rückwärtsdurchlauf sowie die Skripte für Training, Diagramme und Statistik wurden mit Hilfe von Claude (Anthropic) geschrieben. Alle Berechnungen liefen auf dem Computer des Autors.

## Lizenz

GPL-3.0, siehe `LICENSE`.
