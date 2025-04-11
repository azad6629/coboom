import os
import time
import random
import argparse
import yaml
import numpy as np
import torch
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='CoBoom Pre-Training.')
    
    # Training mode and initialization
    parser.add_argument('-tmode', default='pre', choices=['pre', 'down'],
                        help='pre=pre-training or down=downstream transformations')
    parser.add_argument('-init', default=True, action='store_true', 
                        help='Use pre-trained weights (True) or random initialization (False)')
    
    # Model architecture
    parser.add_argument('-arch', default='resnet18', type=str,
                        help='Model architecture (resnet18, resnet50, etc.)')
    
    # Training hyperparameters
    parser.add_argument('-bs', default=64, type=int, 
                        help='Batch size')
    parser.add_argument('-epoch', default=300, type=int,
                        help='Total training epochs')
    parser.add_argument('-lr', default=0.008, type=float,
                        help='Base learning rate')
    parser.add_argument('-mu', default=0.996, type=float,
                        help='Base momentum for target network')
    
    # Model dimensions
    parser.add_argument('-hd', default=4096, type=int,
                        help='Hidden dimension in projection/prediction head')
    parser.add_argument('-od', default=256, type=int,
                        help='Output dimension in projection/prediction head')
    
    # Codebook parameters
    parser.add_argument('-ne', default=1024, type=int,
                        help='Number of codebook embeddings')
    parser.add_argument('-ed', default=512, type=int,
                        help='Size of codebook embedding')
    parser.add_argument('-cc', default=0.25, type=float,
                        help='Commitment cost')
    parser.add_argument('-d', default=0.99, type=float,
                        help='Codebook decay')
    
    # Dataset and device
    parser.add_argument('-dataset', default='NIH14', choices=['NIH14', 'Chex14'],
                        help='Dataset to use')
    parser.add_argument('-gpu', default=0, type=int,
                        help='GPU ID to use')
    
    # Other settings
    parser.add_argument('-ver', default='v1', type=str,
                        help='Version identifier for this run')
    parser.add_argument('-resume', default=False, action='store_true', 
                        help='Resume training from checkpoint')
    parser.add_argument('-seed', default=42, type=int,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def update_config_with_args(config, args):
    """Update config dictionary with command line arguments"""
    # Version and environment settings
    config['ver'] = args.ver
    config['gpu'] = args.gpu
    config['tmode'] = args.tmode
    config['seed'] = args.seed
    
    # Model settings
    config['model_name'] = 'coboom' 
    config['model']['backbone']['type'] = args.arch
    config['model']['backbone']['pretrained'] = args.init
    
    # Dataset and training settings
    config['data']['dataset'] = args.dataset
    config['data']['pre_bs'] = args.bs
    config['optimizer']['total_epochs'] = args.epoch
    config['optimizer']['base_lr'] = args.lr
    config['model']['base_momentum'] = args.mu
    
    # Dimension settings
    config['model']['projection']['hidden_dim'] = args.hd
    config['model']['projection']['output_dim'] = args.od
    config['model']['predictor']['input_dim'] = args.od
    config['model']['predictor']['hidden_dim'] = args.hd
    config['model']['predictor']['output_dim'] = args.od
    
    # Vector quantizer settings
    config['vqconfig']['num_embeddings'] = args.ne
    config['vqconfig']['embedding_dim'] = args.ed
    config['vqconfig']['commitment_cost'] = args.cc
    config['vqconfig']['decay'] = args.d
    
    return config


def main():
    args = parse_args()
    config_file_path = "./config/train_config.yaml"
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config = update_config_with_args(config, args)    
    trainer = Trainer(config)    
    trainer.resume_model(args.resume)
    trainer.logger.info(f"Starting training from epoch {trainer.start_epoch + 1}")
    total_training_time = 0
    trainer.logger.info(f"Total epochs to run: {trainer.total_epochs}")
    
    try:
        for epoch in range(trainer.start_epoch + 1, trainer.total_epochs + 1):
            trainer.logger.info(f"=== Epoch {epoch}/{trainer.total_epochs} ===")            
            start_time = time.time()
            metrics = trainer.train_epoch(epoch)
            epoch_time = time.time() - start_time
            trainer.logger.info(
                f"Epoch {epoch} completed in {epoch_time/60:.2f} minutes | "
                f"Loss: {metrics['loss']:.4f} | CL: {metrics['cl_loss']:.4f} | "
                f"QL: {metrics['q_loss']:.4f} | RL: {metrics['r_loss']:.4f} | PRL: {metrics['perplexity']:.4f}"
            )
            trainer.save_checkpoint(epoch)            
            total_training_time += epoch_time            
            epochs_remaining = trainer.total_epochs - epoch
            time_per_epoch = total_training_time / (epoch - trainer.start_epoch)
            estimated_remaining = epochs_remaining * time_per_epoch
            trainer.logger.info(
                f"Progress: {epoch}/{trainer.total_epochs} epochs | "
                f"Est. remaining: {estimated_remaining/3600:.2f} hours"
            )
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
        trainer.save_checkpoint(epoch)
    
    except Exception as e:
        trainer.logger.error(f"Error during training: {str(e)}")
        import traceback
        trainer.logger.error(traceback.format_exc())
        trainer.save_checkpoint(epoch)
        raise
        
    finally:
        total_training_time_hours = total_training_time / 3600
        trainer.logger.info(f"Total training time: {total_training_time_hours:.2f} hours")

if __name__ == '__main__':
    main()