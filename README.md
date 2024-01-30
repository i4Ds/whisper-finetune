# whisper-finetune

Docker Build:
```
docker build . -t whisper
```

Docker Run:
```
docker run --gpus all -v <PATH-TO-CODE>\whisper-finetune\src\whisper_finetune:/code -v <PATH-TO-CACHE>:/hf --env-file .env whisper python3 code/finetune.py
```

Download the following files to folder and mount folder as `/cache`:
```
https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
```