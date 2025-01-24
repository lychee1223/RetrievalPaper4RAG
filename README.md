
## Setup
```bash
pyenv install 3.11.7
pyenv virtualenv 3.11.7 RetrievalPaper4RAG
pyenv RetrievalPaper4RAG
poetry install
```

## run
```bash
bash bash.sh
```
## parameter
- MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
- EPOCHS=10
- BATCH_SIZE=128
- LEARNING_RATE=1e-5
- MAX_LEN=512
- TEMPERATURE：対照損失における温度 
- IS_USING_MY_SAMLER=False
