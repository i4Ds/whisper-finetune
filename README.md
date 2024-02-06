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

Example:
```
docker run --gpus all -v F:\FHNW\whisper-finetune\src:/code -v F:\FHNW\whisper-finetune\hf:/hf -v F:\FHNW\whisper-finetune\configs:/configs --env-file .env whisper python3 -m whisper_finetune.finetune --config /configs/small.yaml
```
