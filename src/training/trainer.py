"""
Main training logic for diffusion model conversion.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import json
import time
import os

logger = logging.getLogger(__name__)

class DiffusionTrainer:
    """
    Trainer for converting autoregressive models to diffusion models.
    Implements three-stage training:
    1. Attention pattern adaptation
    2. Diffusion objective training
    3. Capability recovery
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=config.get('mixed_precision', 'fp16'),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            cpu=False,
            log_with="wandb" if config.get('use_wandb', False) else None,
            project_dir=config.get('output_dir', './outputs')
        )
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False) and self.accelerator.is_main_process:
            wandb.init(
                project="diffusion-llm-conversion",
                config=config,
                name=config.get('experiment_name', 'diffusion_conversion')
            )
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.training_logs = []
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', './outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
    def train(
        self,
        model,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        optimizer,
        scheduler,
        noise_scheduler,
        tokenizer=None
    ):
        """
        Main training loop with three stages:
        1. Attention pattern adaptation
        2. Diffusion objective training  
        3. Capability recovery
        """
        # Prepare for distributed training
        model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        
        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # Get stage configurations
        stage_configs = self._get_stage_configs()
        
        logger.info(f"Starting training with {len(stage_configs)} stages")
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        
        for stage_idx, stage_config in enumerate(stage_configs, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Stage {stage_idx}: {stage_config['name']}")
            logger.info(f"{'='*60}")
            
            # Configure stage-specific parameters
            self._configure_stage(model, stage_idx, stage_config)
            
            # Train for this stage
            self._train_stage(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                noise_scheduler=noise_scheduler,
                stage_idx=stage_idx,
                stage_config=stage_config,
                tokenizer=tokenizer
            )
            
            # Save checkpoint after each stage
            if self.accelerator.is_main_process:
                self._save_checkpoint(
                    model, optimizer, scheduler, 
                    stage_idx, f"stage_{stage_idx}_final",
                    tokenizer=tokenizer
                )
        
        logger.info("\n🎉 Training completed successfully!")
        
        # Final evaluation
        if eval_dataloader is not None:
            final_eval_loss = self.evaluate(
                model, eval_dataloader, noise_scheduler, stage_idx=3
            )
            logger.info(f"Final evaluation loss: {final_eval_loss:.4f}")
        
        return model
    
    def _get_stage_configs(self) -> List[Dict[str, Any]]:
        """Get configuration for each training stage."""
        return [
            {
                'name': 'Attention Pattern Adaptation',
                'epochs': self.config.get('stage1_epochs', 2),
                'learning_rate_multiplier': 0.5,
                'description': 'Adapt attention patterns for bidirectional processing'
            },
            {
                'name': 'Diffusion Objective Training',
                'epochs': self.config.get('stage2_epochs', 4),
                'learning_rate_multiplier': 1.0,
                'description': 'Train on full diffusion objective'
            },
            {
                'name': 'Capability Recovery',
                'epochs': self.config.get('stage3_epochs', 2),
                'learning_rate_multiplier': 0.3,
                'description': 'Fine-tune to recover language modeling capabilities'
            }
        ]
    
    def _configure_stage(self, model, stage_idx: int, stage_config: Dict[str, Any]):
        """Configure model parameters for each training stage."""
        if stage_idx == 1:
            # Stage 1: Only train attention-related and time embedding parameters
            for name, param in model.named_parameters():
                if any(keyword in name.lower() for keyword in ['attention', 'attn', 'time_embed', 'time_mlp']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        elif stage_idx == 2:
            # Stage 2: Train all diffusion components, keep base model frozen
            for name, param in model.named_parameters():
                if 'base_model' in name:
                    # Only train embeddings in base model
                    if any(keyword in name.lower() for keyword in ['embed', 'wte', 'embed_tokens']):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    # Train all diffusion-specific components
                    param.requires_grad = True
                    
        else:
            # Stage 3: Fine-tune everything
            for param in model.parameters():
                param.requires_grad = True
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Stage {stage_idx}: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M trainable parameters")
    
    def _train_stage(
        self,
        model,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        optimizer,
        scheduler,
        noise_scheduler,
        stage_idx: int,
        stage_config: Dict[str, Any],
        tokenizer=None
    ):
        """Train a single stage."""
        num_epochs = stage_config['epochs']
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            epoch_start_time = time.time()
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Stage {stage_idx} Epoch {epoch+1}/{num_epochs}",
                disable=not self.accelerator.is_main_process
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Stage-specific training step
                loss = self._training_step(
                    model, batch, noise_scheduler, stage_idx, stage_config
                )
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Update weights
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    # Gradient clipping
                    if self.config.get('max_grad_norm', 0) > 0:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(), 
                            self.config['max_grad_norm']
                        )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item()
                self.global_step += 1
                
                # Update progress bar
                avg_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': self.global_step
                })
                
                # Logging
                if self.global_step % self.config.get('logging_steps', 100) == 0:
                    self._log_metrics({
                        f'stage_{stage_idx}_loss': avg_loss,
                        'learning_rate': scheduler.get_last_lr()[0],
                        'global_step': self.global_step,
                        'epoch': epoch + 1,
                        'stage': stage_idx
                    })
                
                # Evaluation
                if (self.global_step % self.config.get('eval_steps', 1000) == 0 and 
                    eval_dataloader is not None):
                    eval_loss = self.evaluate(
                        model, eval_dataloader, noise_scheduler, stage_idx
                    )
                    
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        if self.accelerator.is_main_process:
                            self._save_checkpoint(
                                model, optimizer, scheduler, 
                                stage_idx, "best_checkpoint",
                                tokenizer=tokenizer
                            )
                    
                    model.train()
                
                # Save checkpoint
                if (self.global_step % self.config.get('save_steps', 5000) == 0 and
                    self.accelerator.is_main_process):
                    self._save_checkpoint(
                        model, optimizer, scheduler, 
                        stage_idx, f"checkpoint_{self.global_step}",
                        tokenizer=tokenizer
                    )
            
            # End of epoch logging
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            
            logger.info(f"Stage {stage_idx} Epoch {epoch+1} completed in {epoch_time:.2f}s, avg loss: {avg_epoch_loss:.4f}")
            
            self._log_metrics({
                f'stage_{stage_idx}_epoch_loss': avg_epoch_loss,
                f'stage_{stage_idx}_epoch_time': epoch_time,
                'epoch': epoch + 1,
                'stage': stage_idx
            })
    
    def _training_step(
        self, 
        model, 
        batch: Dict[str, torch.Tensor], 
        noise_scheduler, 
        stage_idx: int,
        stage_config: Dict[str, Any]
    ) -> torch.Tensor:
        """Stage-specific training step."""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.num_timesteps, 
            (batch_size,), device=device
        )
        
        # Add noise
        noisy_ids, mask = noise_scheduler.add_noise(input_ids, timesteps)
        
        # Forward pass
        outputs = model(noisy_ids, timesteps, attention_mask=attention_mask)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # Compute loss based on stage
        if stage_idx == 1:
            # Stage 1: Focus on attention pattern adaptation
            loss = self._attention_adaptation_loss(logits, input_ids, mask, timesteps, noise_scheduler)
        elif stage_idx == 2:
            # Stage 2: Full diffusion objective
            loss = self._diffusion_loss(logits, input_ids, mask, timesteps, noise_scheduler)
        else:
            # Stage 3: Capability recovery with auxiliary tasks
            loss = self._capability_recovery_loss(logits, input_ids, mask, batch, timesteps, noise_scheduler)
        
        return loss
    
    def _attention_adaptation_loss(
        self, 
        logits: torch.Tensor, 
        target_ids: torch.Tensor, 
        mask: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scheduler
    ) -> torch.Tensor:
        """Loss for attention pattern adaptation stage."""
        # Standard diffusion loss with reduced weight
        diffusion_loss = self._diffusion_loss(logits, target_ids, mask, timesteps, noise_scheduler)
        
        # Additional regularization to encourage bidirectional attention
        # This is a simplified version - more sophisticated methods could be used
        return diffusion_loss * 0.5
    
    def _diffusion_loss(
        self, 
        logits: torch.Tensor, 
        target_ids: torch.Tensor, 
        mask: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scheduler
    ) -> torch.Tensor:
        """Masked diffusion loss with timestep weighting."""
        # Only compute loss on masked positions
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Flatten for loss computation
        masked_logits = logits[mask]
        masked_targets = target_ids[mask]
        
        # Compute cross entropy loss
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')
        
        # Apply timestep-dependent weighting
        weights = noise_scheduler.get_loss_weight(timesteps)
        # Expand weights to match mask shape
        expanded_weights = weights.unsqueeze(1).expand_as(mask)[mask]
        
        # Apply weights and average
        weighted_loss = (loss * expanded_weights).mean()
        
        return weighted_loss
    
    def _capability_recovery_loss(
        self, 
        logits: torch.Tensor, 
        target_ids: torch.Tensor, 
        mask: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        noise_scheduler
    ) -> torch.Tensor:
        """Loss for capability recovery stage."""
        # Primary diffusion loss
        diffusion_loss = self._diffusion_loss(logits, target_ids, mask, timesteps, noise_scheduler)
        
        # Auxiliary language modeling loss (on non-masked tokens)
        non_masked = ~mask
        if non_masked.sum() > 0:
            # Compute standard LM loss on non-masked tokens
            lm_logits = logits[non_masked]
            lm_targets = target_ids[non_masked]
            lm_loss = F.cross_entropy(lm_logits, lm_targets, reduction='mean')
            
            # Combine losses
            total_loss = diffusion_loss + 0.1 * lm_loss
        else:
            total_loss = diffusion_loss
        
        return total_loss
    
    def evaluate(
        self,
        model,
        eval_dataloader: DataLoader,
        noise_scheduler,
        stage_idx: int
    ) -> float:
        """Evaluate the model."""
        model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_main_process):
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                batch_size = input_ids.shape[0]
                device = input_ids.device
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.num_timesteps, 
                    (batch_size,), device=device
                )
                
                # Add noise
                noisy_ids, mask = noise_scheduler.add_noise(input_ids, timesteps)
                
                # Forward pass
                outputs = model(noisy_ids, timesteps, attention_mask=attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Compute loss
                loss = self._diffusion_loss(logits, input_ids, mask, timesteps, noise_scheduler)
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        
        # Log evaluation metrics
        self._log_metrics({
            f'stage_{stage_idx}_eval_loss': avg_loss,
            'global_step': self.global_step
        })
        
        logger.info(f"Evaluation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        stage_idx: int,
        checkpoint_name: str,
        tokenizer=None
    ):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_dir)
        
        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'stage': stage_idx,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config
        }
        
        torch.save(training_state, checkpoint_dir / 'training_state.pt')
        
        # Save optimizer and scheduler states
        torch.save(optimizer.state_dict(), checkpoint_dir / 'optimizer.pt')
        torch.save(scheduler.state_dict(), checkpoint_dir / 'scheduler.pt')
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics."""
        self.training_logs.append(metrics)
        
        if self.config.get('use_wandb', False) and self.accelerator.is_main_process:
            wandb.log(metrics)
        
        # Save logs to file
        if self.accelerator.is_main_process:
            with open(self.output_dir / 'training_logs.json', 'w') as f:
                json.dump(self.training_logs, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load training state
        training_state = torch.load(checkpoint_dir / 'training_state.pt')
        self.global_step = training_state['global_step']
        self.best_eval_loss = training_state['best_eval_loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}, global_step: {self.global_step}")
        
        return training_state