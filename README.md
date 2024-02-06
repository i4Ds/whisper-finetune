# whisper-finetune

Docker Build:
```
docker build . -t whisper
```

Fill in .env file with your credentials and paths, see .env-template for reference.

Docker Run:
```
docker run --gpus all -v <PATH_TO_CODE>\whisper-finetune\src:/code -v <PATH_TO_HF_CACHE>:/hf -v <PATH_TO_CODE>\configs:/configs --env-file .env whisper python3 -m whisper_finetune.finetune --config /configs/small.yaml
```

Docker Run Example:
```
docker run --gpus all -v F:\FHNW\whisper-finetune\src:/code -v F:\FHNW\whisper-finetune\hf:/hf -v F:\FHNW\whisper-finetune\configs:/configs --env-file .env whisper python3 -m whisper_finetune.finetune --config /configs/small.yaml
```
