import os
from pathlib import Path
import yaml
import torch
import os

from trainer.trainer_VQ import VQTrainer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def run_task(config):
    trainer = VQTrainer(config)
    trainer.resume_model(model_path=None)
    start_epoch = trainer.start_epoch

    for epoch in range(start_epoch + 1, trainer.total_epochs + 1):
        trainer.train_epoch(epoch)
        trainer.save_checkpoint(epoch)
        
def main():
    with open('./config/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    run_task(config)

if __name__ == "__main__":
      main()