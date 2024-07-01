#!/usr/bin/env python
# coding: utf-8
# %%

# %%


#-*- coding:utf-8 -*-
import os
from pathlib import Path
import yaml
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# %%


from trainer.T1byol_trainer_VQL import BYOLTrainer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# %%


def run_task(config):
    trainer = BYOLTrainer(config)
    trainer.resume_model(model_path=None)
    start_epoch = trainer.start_epoch

    for epoch in range(start_epoch + 1, trainer.total_epochs + 1):
#         trainer.save_checkpoint(epoch)
        trainer.train_epoch(epoch)
        trainer.save_checkpoint(epoch)


# %%
def main():
    with open('/workspace/data/VQSSL/config/train_config_mimic.yaml', 'r') as f:
        config = yaml.safe_load(f)
    run_task(config)

if __name__ == "__main__":
      main()


# %%





# %%




