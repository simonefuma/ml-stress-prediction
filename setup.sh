# Controlla se l'ambiente virtuale esiste già
if [ -d ".venv" ]; then
  echo "L'ambiente virtuale esiste già."
else
  # Creazione dell'ambiente virtuale se non esiste
  echo "L'ambiente virtuale non esiste. Creazione dell'ambiente..."
  python3 -m venv .venv --prompt ml-stress-prediction
fi

# Attivazione dell'ambiente virtuale
echo "Attivazione dell'ambiente virtuale..."
source .venv/bin/activate

# Installazione delle dipendenze da requirements.txt
if [ -f "requirements.txt" ]; then
  echo "Installazione delle dipendenze da requirements.txt..."
  pip install -r requirements.txt
else
  echo "File requirements.txt non trovato!"
  deactivate
  exit 1
fi

# Fine del processo
echo ""
echo "Installazione completata. Ora l'ambiente virtuale è attivo."
echo "Quando hai finito, puoi disattivare l'ambiente virtuale con 'deactivate'."