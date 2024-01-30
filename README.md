# whisper-finetune

Docker Build:
```
docker build . -t whisper
```

Fill in .env file with your credentials and paths, see .env-template for reference.

Docker Run:
```
docker run --gpus all -v <PATH-TO-CODE>\whisper-finetune\src\whisper_finetune:/code -v <PATH-TO-CACHE>:/hf --env-file .env whisper python3 code/finetune.py
```
