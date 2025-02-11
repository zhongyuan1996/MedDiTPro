import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import logging
import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
from datetime import datetime
import torch.nn.functional as F

from seqStyleDataset import EHRDataloader, create_dataloader
from config_mimic3 import MIMIC3_ModelConfig, MIMIC3_TrainingConfig
from config_mimic4_icd9 import MIMIC4_icd9_ModelConfig, MIMIC4_icd9_TrainingConfig
from config_eicu import EICU_ModelConfig, EICU_TrainingConfig
from config_breast import BREAST_ModelConfig, BREAST_TrainingConfig
from model import EHRModel
from evaluator import InVisitEvaluator
import math
import os
import argparse
from typing import Union

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class FocalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.alpha
        self.gamma = config.gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class Runner:
    def __init__(self,
                model_config: Union[MIMIC3_ModelConfig, MIMIC4_icd9_ModelConfig, EICU_ModelConfig, BREAST_ModelConfig],
                training_config: Union[MIMIC3_TrainingConfig, MIMIC4_icd9_TrainingConfig, EICU_TrainingConfig, BREAST_TrainingConfig],
                resume_from=None,
                ablation=None,
                timed=False):
        self.model_config = model_config
        self.training_config = training_config
        self.current_step = 0
        self.resume_from = resume_from
        self.ablation = ablation
        self.timed = timed
        self.dataset_name = ('mimic3' if isinstance(training_config, MIMIC3_TrainingConfig) 
                     else 'mimic4_icd9' if isinstance(training_config, MIMIC4_icd9_TrainingConfig) 
                     else 'eicu' if isinstance(training_config, EICU_TrainingConfig)
                     else 'breast' if isinstance(training_config, BREAST_TrainingConfig)
                     else None)
        assert self.dataset_name is not None, "Unsupported training configuration type"

        # Modify paths for ablation studies
        if ablation:
            # Add ablation info to experiment name
            self.training_config.experiment_name = f"{self.training_config.experiment_name}_{ablation}"
            
            # Create ablation-specific directories
            self.training_config.log_dir = os.path.join(self.training_config.log_dir, f"ablation_{self.dataset_name}_{ablation}")
            self.training_config.checkpoint_dir = os.path.join(self.training_config.checkpoint_dir, f"ablation_{self.dataset_name}_{ablation}")
            self.training_config.tensorboard_dir = os.path.join(self.training_config.tensorboard_dir, f"ablation_{self.dataset_name}_{ablation}")
            
            # Log ablation setup
            print(f"\nAblation Study Setup:")
            print(f"Type: {ablation}")
            print(f"Log directory: {self.training_config.log_dir}")
            print(f"Checkpoint directory: {self.training_config.checkpoint_dir}")
            print(f"Tensorboard directory: {self.training_config.tensorboard_dir}")

        
        # Set random seeds
        self.set_seed(training_config.seed)
        
        # Setup logging
        self.setup_logging()
        
        # Log ablation information
        if self.ablation:
            logging.info(f"Running ablation study: {self.ablation}")
        
        # Initialize model and move to device
        self.device = torch.device(training_config.device)
        self.model = EHRModel(model_config).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            betas=training_config.adam_betas,
            eps=training_config.adam_epsilon,
            weight_decay=training_config.weight_decay
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(
            os.path.join(training_config.tensorboard_dir, training_config.experiment_name)
        )
        
        # Setup mixed precision if enabled
        self.scaler = torch.amp.GradScaler('cuda') if training_config.use_amp else None
        
        # Track best model
        self.best_val_loss = float('inf')
        self.best_median_lpl = float('inf')
        self.best_median_mpl = float('inf')
        self.patience_counter = 0
        self.start_epoch = 0

        if resume_from:
            # Modify resume path for ablation if needed
            if ablation and "ablation_" not in resume_from:
                resume_from = os.path.join(f"ablation_{ablation}", resume_from)
            self.load_checkpoint(resume_from)
            
            # Extract epoch number from checkpoint filename if it's a periodic checkpoint
            if 'checkpoint_epoch_' in resume_from:
                try:
                    self.start_epoch = int(resume_from.split('_')[-1].split('.')[0]) + 1
                except ValueError:
                    logging.warning("Could not determine epoch number from checkpoint filename")
            
            logging.info(f"Resuming training from epoch {self.start_epoch}")

        
    def setup_logging(self):
        """Setup logging configuration"""
        # Ensure the log directory exists
        log_dir = Path(self.training_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_dir / f"{self.training_config.experiment_name}.log"),
                logging.StreamHandler()
            ]
        )
        
    def set_seed(self, seed):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def sample_timesteps(self, batch_size):
        """Sample random timesteps for each modality"""
        return {
            'diag': torch.rand(batch_size, device=self.device),
            'drug': torch.rand(batch_size, device=self.device),
            'proc': torch.rand(batch_size, device=self.device),
            'lab': torch.rand(batch_size, device=self.device)
        }
    
    def create_dataloaders(self):
        """
        Create train and validation dataloaders based on config type
        Handles both MIMIC-III and MIMIC-IV ICD9 configurations
        """
        if self.training_config.debug:
            # Use toy dataset for both train and val in debug mode
            if isinstance(self.training_config, MIMIC3_TrainingConfig):
                data_path = self.training_config.mimiciii_toy_path
                code_to_index_path = self.training_config.mimiciii_mapping_path
            elif isinstance(self.training_config, MIMIC4_icd9_TrainingConfig):
                data_path = self.training_config.mimiciv_toy_path
                code_to_index_path = self.training_config.mimiciv_mapping_path
            elif isinstance(self.training_config, EICU_TrainingConfig):
                data_path = self.training_config.eicu_toy_path
                code_to_index_path = self.training_config.eicu_mapping_path
            elif isinstance(self.training_config, BREAST_TrainingConfig):
                data_path = self.training_config.breast_toy_path
                code_to_index_path = self.training_config.breast_mapping_path
            else:
                raise ValueError("Unsupported training configuration type")

            train_loader = create_dataloader(
                data_path=data_path,
                code_to_index_path=code_to_index_path,
                batch_size=self.training_config.batch_size,
                max_seq_length=self.model_config.max_position_embeddings,
                num_workers=self.training_config.num_workers,
                mask_prob=self.training_config.mask_prob
            )
            val_loader = train_loader
            test_loader = train_loader
        
        else:
            # Regular training mode
            if isinstance(self.training_config, MIMIC3_TrainingConfig):
                train_loader = create_dataloader(
                    data_path=self.training_config.mimiciii_train_path,
                    code_to_index_path=self.training_config.mimiciii_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                val_loader = create_dataloader(
                    data_path=self.training_config.mimiciii_val_path,
                    code_to_index_path=self.training_config.mimiciii_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                test_loader = create_dataloader(
                    data_path=self.training_config.mimiciii_test_path,
                    code_to_index_path=self.training_config.mimiciii_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
            
            elif isinstance(self.training_config, MIMIC4_icd9_TrainingConfig):
                train_loader = create_dataloader(
                    data_path=self.training_config.mimiciv_train_path,
                    code_to_index_path=self.training_config.mimiciv_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                val_loader = create_dataloader(
                    data_path=self.training_config.mimiciv_val_path,
                    code_to_index_path=self.training_config.mimiciv_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                test_loader = create_dataloader(
                    data_path=self.training_config.mimiciv_test_path,
                    code_to_index_path=self.training_config.mimiciv_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )

            elif isinstance(self.training_config, EICU_TrainingConfig):
                train_loader = create_dataloader(
                    data_path=self.training_config.eicu_train_path,
                    code_to_index_path=self.training_config.eicu_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                val_loader = create_dataloader(
                    data_path=self.training_config.eicu_val_path,
                    code_to_index_path=self.training_config.eicu_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                test_loader = create_dataloader(
                    data_path=self.training_config.eicu_test_path,
                    code_to_index_path=self.training_config.eicu_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
            elif isinstance(self.training_config, BREAST_TrainingConfig):
                train_loader = create_dataloader(
                    data_path=self.training_config.breast_train_path,
                    code_to_index_path=self.training_config.breast_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                val_loader = create_dataloader(
                    data_path=self.training_config.breast_val_path,
                    code_to_index_path=self.training_config.breast_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                test_loader = create_dataloader(
                    data_path=self.training_config.breast_test_path,
                    code_to_index_path=self.training_config.breast_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
            
            else:
                raise ValueError("Unsupported training configuration type")

        return train_loader, val_loader, test_loader
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_diff_loss = 0
        total_pred_loss = 0
        num_batches = len(dataloader)
        focal_loss = FocalLoss(self.training_config)
        
        with tqdm(dataloader, desc=f'Epoch {epoch}', total=num_batches) as pbar:
            for step, batch in enumerate(pbar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                modality_mask = batch['modality_mask'].to(self.device)
                modality_indices = batch['modality_indices'].to(self.device)
                mlm_labels = batch['mlm_labels'].to(self.device)

                if self.timed:
                    time_gaps = batch['time_gaps'].to(self.device)
                
                timesteps = self.sample_timesteps(input_ids.size(0))
                
                with torch.amp.autocast(device_type='cuda', enabled=bool(self.scaler)):
                    if self.timed:
                        predictions, diffusion_loss, time_loss = self.model.time_forward(
                            input_ids=input_ids,
                            timegaps=time_gaps,
                            seq_mask=attention_mask,
                            modality_mask=modality_mask,
                            modality_indices=modality_indices,
                            timesteps=timesteps,
                            ablation=self.ablation
                        )
                    else:
                        predictions, diffusion_loss = self.model(
                            input_ids=input_ids,
                            seq_mask=attention_mask,
                            modality_mask=modality_mask,
                            modality_indices=modality_indices,
                            timesteps=timesteps,
                            ablation=self.ablation
                        )
                    
                    # Calculate prediction loss for each modality
                    pred_loss = 0
                    for modality in ['diag', 'drug', 'proc', 'lab']:
                        pred_dict = predictions[modality]
                        logits = pred_dict['logits']
                        valid_mask = pred_dict['mask']
                        
                        if valid_mask.sum() > 0:
                            targets = torch.where(
                                mlm_labels != -100,
                                mlm_labels,
                                input_ids
                            )
                            
                            batch_idx, seq_idx = torch.where(valid_mask)
                            flat_targets = targets[batch_idx, seq_idx]
                            special_token_values = set(self.model_config.special_tokens.values())
                            valid_target_mask = ~torch.tensor([t.item() in special_token_values for t in flat_targets],
                                                            device=flat_targets.device)
                            
                            if valid_target_mask.any():
                                valid_batch_idx = batch_idx[valid_target_mask]
                                valid_seq_idx = seq_idx[valid_target_mask]
                                flat_logits = logits[valid_batch_idx, valid_seq_idx]
                                flat_targets = flat_targets[valid_target_mask]
                                
                                vocab_offset = {
                                    'diag': len(self.model_config.special_tokens),
                                    'drug': len(self.model_config.special_tokens) + self.model_config.diag_vocab_size,
                                    'proc': len(self.model_config.special_tokens) + self.model_config.diag_vocab_size + self.model_config.drug_vocab_size,
                                    'lab': len(self.model_config.special_tokens) + self.model_config.diag_vocab_size + self.model_config.drug_vocab_size + self.model_config.proc_vocab_size
                                }
                            

                                flat_targets = flat_targets - vocab_offset[modality]
                                
                                assert (flat_targets >= 0).all() and (flat_targets < logits.size(-1)).all(), \
                                    f"Target values outside valid range for {modality}."
                                
                                pred_loss += focal_loss(flat_logits, flat_targets)
                    
                    weighted_pred_loss = pred_loss * self.training_config.pred_weight
                    weighted_diff_loss = diffusion_loss * self.training_config.diff_weight
                    
                    if self.timed and not torch.isnan(time_loss) and not torch.isinf(time_loss):
                        weighted_time_loss = time_loss * self.training_config.time_loss_weight
                        loss = weighted_pred_loss + weighted_diff_loss + weighted_time_loss
                        total_time_loss += weighted_time_loss.item()
                    else:
                        loss = weighted_pred_loss + weighted_diff_loss
                    
                    # Rest of training step remains the same
                    if self.scaler:
                        scaled_loss = self.scaler.scale(loss / self.training_config.gradient_accumulation_steps)
                        scaled_loss.backward()
                    else:
                        (loss / self.training_config.gradient_accumulation_steps).backward()
                    
                    if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                            self.optimizer.step()
                        
                        self.optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    total_diff_loss += weighted_diff_loss.item()
                    total_pred_loss += weighted_pred_loss.item() if torch.is_tensor(weighted_pred_loss) else weighted_pred_loss # added for breast dataset

                    
                    pbar_postfix = {
                        'loss': loss.item(),
                        'diff_loss': weighted_diff_loss.item(),
                        'pred_loss': weighted_pred_loss.item() if torch.is_tensor(weighted_pred_loss) else weighted_pred_loss #same reason as above
                    }
                    if self.timed and not torch.isnan(time_loss) and not torch.isinf(time_loss):
                        pbar_postfix['time_loss'] = weighted_time_loss.item()
                    
                    pbar.set_postfix(pbar_postfix)
                    
                    if step % self.training_config.log_steps == 0:
                        self.writer.add_scalar('train/loss', loss.item(), epoch * num_batches + step)
                        self.writer.add_scalar('train/diffusion_loss', weighted_diff_loss.item(), epoch * num_batches + step)
                        self.writer.add_scalar(
                            'train/prediction_loss', 
                            weighted_pred_loss.item() if torch.is_tensor(weighted_pred_loss) else weighted_pred_loss, # same reason as above
                            epoch * num_batches + step
                        )
                        if self.timed and not torch.isnan(time_loss) and not torch.isinf(time_loss):
                            self.writer.add_scalar(
                                'train/time_loss',
                                weighted_time_loss.item() if torch.is_tensor(weighted_time_loss) else weighted_time_loss, # same reason as above
                                epoch * num_batches + step
                            )

            self.current_step += 1
            metrics = {
                'loss': total_loss / num_batches,
                'diff_loss': total_diff_loss / num_batches,
                'pred_loss': total_pred_loss / num_batches
            }
            if self.timed:
                metrics['time_loss'] = total_time_loss / num_batches
                
            return metrics
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate the model using InVisitEvaluator"""
        evaluator = InVisitEvaluator(self.model, self.device)
        metrics = evaluator.evaluate(dataloader)
        
        # Log each metric to tensorboard
        for metric_name, value in metrics.items():
            if not math.isinf(value):
                self.writer.add_scalar(f'val/{metric_name}', value, self.current_step)
        
        # Split metrics into lpl and mpl for each modality
        lpl_metrics = {
            modality: metrics[f'lpl_{modality}'] 
            for modality in ['diag', 'drug', 'proc', 'lab']
        }
        mpl_metrics = {
            modality: metrics[f'mpl_{modality}']
            for modality in ['diag', 'drug', 'proc', 'lab']
        }
        
        # Calculate median scores for logging
        valid_lpl = [v for v in lpl_metrics.values() if not math.isinf(v)]
        valid_mpl = [v for v in mpl_metrics.values() if not math.isinf(v)]
        
        if valid_lpl:
            median_lpl = np.median(valid_lpl)
            self.writer.add_scalar('val/median_lpl', median_lpl, self.current_step)
        
        if valid_mpl:
            median_mpl = np.median(valid_mpl)
            self.writer.add_scalar('val/median_mpl', median_mpl, self.current_step)
        
        return lpl_metrics, mpl_metrics
    
    def train(self):
        """Main training loop"""
        if self.ablation:
            logging.info(f"Starting ablation study training ({self.ablation}) from epoch {self.start_epoch}...")
        else:
            logging.info(f"Starting training from epoch {self.start_epoch}...")
        start_time = time.time()
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders()

        # Setup learning rate scheduler
        num_training_steps = len(train_loader) * self.training_config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Skip scheduler steps if resuming
        if self.start_epoch > 0:
            for _ in range(self.start_epoch * len(train_loader)):
                scheduler.step()
        
        for epoch in range(self.training_config.num_epochs):
            # Train one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            logging.info(f"Epoch {epoch} training metrics: {train_metrics}")
            
            # Evaluate
            lpl_metrics, mpl_metrics = self.evaluate(val_loader)
            
            # Calculate median scores
            curr_median_lpl = np.median([v for v in lpl_metrics.values() if not math.isinf(v)])
            curr_median_mpl = np.median([v for v in mpl_metrics.values() if not math.isinf(v)])
            
            logging.info(f"Epoch {epoch} validation LPL per modality: {lpl_metrics}")
            logging.info(f"Epoch {epoch} validation MPL per modality: {mpl_metrics}")
            logging.info(f"Epoch {epoch} median LPL: {curr_median_lpl:.4f}")
            logging.info(f"Epoch {epoch} median MPL: {curr_median_mpl:.4f}")
            
            # Early stopping check - save if either metric improves
            improved = False
            if curr_median_lpl < self.best_median_lpl:
                self.best_median_lpl = curr_median_lpl
                improved = True
                
            if curr_median_mpl < self.best_median_mpl:
                self.best_median_mpl = curr_median_mpl
                improved = True
                
            if improved:
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.training_config.patience:
                logging.info("Early stopping triggered")
                break
                
            # Save periodic checkpoint
            if (epoch + 1) % self.training_config.save_steps == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Step the scheduler
            scheduler.step()
        
        training_time = time.time() - start_time
        logging.info(f"Training finished in {training_time:.2f} seconds")

        # Test set evaluation using best model
        logging.info("Starting test set evaluation...")
        
        # Load best model saved during training
        logging.info("Loading best model for test set evaluation...")
        self.load_checkpoint('best_model.pt')
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Evaluate on test set with no gradients
        logging.info("Evaluating on test set...")
        with torch.no_grad():
            test_lpl_metrics, test_mpl_metrics = self.evaluate(test_loader)
        
        # Calculate and log test set median scores
        test_median_lpl = np.median([v for v in test_lpl_metrics.values() if not math.isinf(v)])
        test_median_mpl = np.median([v for v in test_mpl_metrics.values() if not math.isinf(v)])
        
        logging.info("Test Set Results:")
        logging.info(f"Test LPL per modality: {test_lpl_metrics}")
        logging.info(f"Test MPL per modality: {test_mpl_metrics}")
        logging.info(f"Test median LPL: {test_median_lpl:.4f}")
        logging.info(f"Test median MPL: {test_median_mpl:.4f}")
        
        # Save test results with ablation information
        test_results = {
            'lpl_metrics': test_lpl_metrics,
            'mpl_metrics': test_mpl_metrics,
            'median_lpl': test_median_lpl,
            'median_mpl': test_median_mpl,
            'ablation': self.ablation,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        
        # Save to ablation-specific results file
        results_filename = 'ablation_test_results.pt' if self.ablation else 'test_results.pt'
        results_filename = 'timed_test_results.pt' if self.timed else results_filename
        test_results_path = Path(self.training_config.checkpoint_dir) / results_filename
        torch.save(test_results, test_results_path)
        logging.info(f"Saved test results to {test_results_path}")
        
        self.writer.close()
        logging.info("Training and evaluation completed.")

    def save_checkpoint(self, filename):
        """Save model checkpoint with comprehensive training state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'current_epoch': self.current_step,
            'best_median_lpl': getattr(self, 'best_median_lpl', float('inf')),
            'best_median_mpl': getattr(self, 'best_median_mpl', float('inf')),
            'patience_counter': getattr(self, 'patience_counter', 0),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'ablation': self.ablation,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        
        # Create directories if they don't exist
        save_path = Path(self.training_config.checkpoint_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
        logging.info(f"Saved checkpoint to {save_path}")
        if self.ablation:
            logging.info(f"Ablation type: {self.ablation}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint and restore training state"""
        load_path = Path(self.training_config.checkpoint_dir) / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.current_step = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_median_lpl = checkpoint.get('best_median_lpl', float('inf'))
        self.best_median_mpl = checkpoint.get('best_median_mpl', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        # Restore scaler if it exists
        if self.scaler and checkpoint.get('scaler'):
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        logging.info(f"Loaded checkpoint from {load_path}")

def get_config(dataset_name: str):
    """
    Get the appropriate model and training configs based on dataset name
    
    Args:
        dataset_name: Name of the dataset ('mimic3' or 'mimic4_icd9')
    
    Returns:
        tuple: (model_config, training_config)
    """
    if dataset_name.lower() == 'mimic3':
        return MIMIC3_ModelConfig(), MIMIC3_TrainingConfig()
    elif dataset_name.lower() == 'mimic4_icd9':
        return MIMIC4_icd9_ModelConfig(), MIMIC4_icd9_TrainingConfig()
    elif dataset_name.lower() == 'eicu':
        return EICU_ModelConfig(), EICU_TrainingConfig()
    elif dataset_name.lower() == 'breast':
        return BREAST_ModelConfig(), BREAST_TrainingConfig()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from: 'mimic3', 'mimic4_icd9', 'eicu', 'breast'")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Train EHR model on specified dataset')
    parser.add_argument('--dataset', type=str, 
                      choices=['mimic3', 'mimic4_icd9','eicu','breast'],
                      default='mimic3',
                      help='Dataset to use for training')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    parser.add_argument('--ablation', type=str, default=None, choices=['ab1_shared', 'ab2_modality', 'ab3_cross', 'ab4_diffusion'])
    parser.add_argument('--timed', default=False, help='Legacy argument for training with time information, deprecated')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get appropriate configs based on dataset name
    model_config, training_config = get_config(args.dataset)
    
    # Print training setup
    print(f"\nTraining Setup:")
    print(f"Dataset: {args.dataset}")
    print(f"Resume from: {args.resume_from if args.resume_from else 'No checkpoint'}")
    print(f"Experiment name: {training_config.experiment_name}")
    print(f"Debug mode: {'Enabled' if training_config.debug else 'Disabled'}")
    print(f"Training model with time information: {args.timed}")
    
    # Create runner
    runner = Runner(model_config, training_config, resume_from=args.resume_from, ablation=args.ablation, timed=args.timed)
    
    # Start training
    runner.train()
