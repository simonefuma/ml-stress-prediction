# ml-stress-prediction
Metodi di machine learning per la predizione dello stato di stress indotto da esperimenti su cavie da laboratorio.

## Prerequisiti

Assicurati di avere installato **Python3** puoi controllare la tua versione usando il seguente comando:
```
python3 --version
```
Se non è installato puoi installarlo dal sito ufficiale: <a href="https://www.python.org/downloads/" target="_blank">Python Download</a>.

## Setup dell'ambiente virtuale

Per configurare l'ambiente del progetto, hai due opzioni:

### Opzione 1: Eseguire lo Script di Setup

Puoi eseguire uno script di setup che configura automaticamente l'ambiente virtuale e installa tutte le dipendenze necessarie. Esegui il seguente comando:

- **MacOS/Linux:**

Per renderlo eseguibile:
```
chmod +x setup.sh
```
Per Lanciarlo:
```
./setup.sh
```
### Opzione 2: Setup Manuale

- **MacOS/Linux:**

Utilizza il modulo `venv` integrato in Python3 per creare l'ambiente virtuale. Da dentro la cartella del progetto, esegui il seguente comando:
```
python3 -m venv .venv --promt ml-stress-prediction
```

Per attivare l'ambiente virtuale appena creato usa questo comando:
```
source .venv/bin/activate
```

Ora che l'ambiente virtuale è attivo, puoi installare tutte le librerie necessarie per il progetto. Le dipendenze sono elencate nel file `requirements.txt`. Usa il seguente comando per installarle:
```
pip install -r requirements.txt
```

Quando hai finito di lavorare, puoi disattivare l'ambiente virtuale con il comando:
```
deactivate
```

## Utilizzo di Jupitext

Jupytext permette, tra le altre cose, di convertire qualsiasi notebook Jupyter in un file `.py` costituito solo dal contenuto effettivo del notebook e privo di tutti i metadati che i file `.ipynb` si trascinano dietro. In questo modo, versioniamo solo ciò che è strettamente necessario.

Si deve convertire il file `.py` in `.ipynb` per visualizzare il notebook con la sua ultima versione, ed è importante riconvertirlo poi per poter versionare le modifiche che abbiamo fatto.

- Convertire da `.py` a `.ipynb`
```
jupytext --to notebook notebook.py
```

- Convertire da `.ipynb` a `.py`
```
jupytext --to py notebook.ipynb 
```