"""
Evaluation metrics for diffusion language models.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DiffusionEvaluator:
    """Evaluator for diffusion language models."""
    
    def __init__(self, tokenizer, device='cuda'):
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize metrics
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        except ImportError:
            logger.warning("rouge_score not available")
            self.rouge_scorer = None
    
    def compute_perplexity(self, model, dataloader, noise_scheduler, max_batches=None):
        """
        Compute perplexity for diffusion models using reconstruction loss.
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        batches_processed = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if max_batches and batches_processed >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                batch_size = input_ids.shape[0]
                
                # Sample multiple timesteps for better estimate
                timestep_losses = []
                for t in range(0, noise_scheduler.num_timesteps, max(1, noise_scheduler.num_timesteps // 10)):
                    timesteps = torch.full((batch_size,), t, device=self.device)
                    noisy_ids, mask = noise_scheduler.add_noise(input_ids, timesteps)
                    
                    if mask.sum() == 0:
                        continue
                    
                    outputs = model(noisy_ids, timesteps, attention_mask)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Compute loss only on masked positions
                    masked_logits = logits[mask]
                    masked_targets = input_ids[mask]
                    
                    loss = torch.nn.functional.cross_entropy(
                        masked_logits, masked_targets, reduction='sum'
                    )
                    timestep_losses.append(loss.item())
                    total_tokens += mask.sum().item()
                
                if timestep_losses:
                    total_loss += sum(timestep_losses) / len(timestep_losses)
                
                batches_processed += 1
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def generate_samples(
        self,
        model,
        prompts: List[str],
        noise_scheduler,
        max_length: int = 100,
        num_steps: int = 20,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> List[str]:
        """
        Generate text samples using diffusion sampling.
        """
        model.eval()
        generated_texts = []
        
        for prompt in prompts:
            try:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors='pt', 
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                prompt_len = inputs['input_ids'].shape[1]
                
                # Initialize sequence with masked tokens
                seq_len = max(max_length, prompt_len + 50)
                input_ids = torch.full(
                    (1, seq_len), 
                    self.tokenizer.mask_token_id, 
                    device=self.device
                )
                
                # Set prompt tokens
                input_ids[:, :prompt_len] = inputs['input_ids']
                
                # Diffusion sampling loop
                timesteps = torch.linspace(
                    noise_scheduler.num_timesteps - 1, 
                    0, 
                    num_steps, 
                    device=self.device
                ).long()
                
                with torch.no_grad():
                    for t in timesteps:
                        timestep_batch = t.expand(1)
                        
                        outputs = model(input_ids, timestep_batch)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        # Apply temperature
                        if temperature != 1.0:
                            logits = logits / temperature
                        
                        # Top-k sampling
                        if top_k > 0:
                            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                            logits[indices_to_remove] = float('-inf')
                        
                        # Convert to probabilities
                        probs = torch.softmax(logits, dim=-1)
                        
                        # Only update masked positions beyond the prompt
                        mask = (input_ids == self.tokenizer.mask_token_id) & (
                            torch.arange(seq_len, device=self.device)[None, :] >= prompt_len
                        )
                        
                        if mask.any():
                            # Sample for masked positions
                            masked_probs = probs[mask]
                            sampled_ids = torch.multinomial(masked_probs, num_samples=1).squeeze(-1)
                            input_ids[mask] = sampled_ids
                
                # Decode generated text
                generated_ids = input_ids[0, prompt_len:]
                
                # Remove mask tokens and decode
                valid_ids = generated_ids[generated_ids != self.tokenizer.mask_token_id]
                if len(valid_ids) > 0:
                    generated_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                    full_text = prompt + generated_text
                else:
                    full_text = prompt
                
                generated_texts.append(full_text)
                
            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                generated_texts.append(prompt + " [GENERATION FAILED]")
        
        return generated_texts
    
    def compute_generation_metrics(
        self,
        model,
        noise_scheduler,
        test_prompts: Optional[List[str]] = None,
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute various generation quality metrics."""
        
        if test_prompts is None:
            test_prompts = [
                "The future of artificial intelligence is",
                "Once upon a time in a distant galaxy",
                "The key to solving climate change involves",
                "In the year 2050, humanity will"
            ]
        
        # Generate samples
        generated_texts = self.generate_samples(
            model, test_prompts, noise_scheduler,
            max_length=100, num_steps=20
        )
        
        metrics = {
            "num_prompts": len(test_prompts),
            "generated_samples": [
                {"prompt": p, "generation": g}
                for p, g in zip(test_prompts, generated_texts)
            ]
        }
        
        # Compute basic statistics
        generation_lengths = [
            len(self.tokenizer.encode(g.split(p, 1)[-1])) 
            for p, g in zip(test_prompts, generated_texts)
        ]
        
        metrics.update({
            "avg_generation_length": np.mean(generation_lengths),
            "std_generation_length": np.std(generation_lengths),
            "min_generation_length": np.min(generation_lengths),
            "max_generation_length": np.max(generation_lengths)
        })
        
        # Compute ROUGE scores if reference texts provided
        if reference_texts and self.rouge_scorer:
            rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
            
            for gen, ref in zip(generated_texts, reference_texts):
                scores = self.rouge_scorer.score(gen, ref)
                rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
                rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
                rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
            
            metrics.update({
                "rouge1_mean": np.mean(rouge_scores["rouge1"]),
                "rouge2_mean": np.mean(rouge_scores["rouge2"]),
                "rougeL_mean": np.mean(rouge_scores["rougeL"])
            })
        
        return metrics
    
    def quick_evaluation(self, model, noise_scheduler, num_samples: int = 4) -> Dict[str, Any]:
        """Quick evaluation with basic metrics."""
        
        test_prompts = [
            "The capital of France is",
            "In machine learning, neural networks",
            "Climate change is caused by",
            "The Python programming language"
        ][:num_samples]
        
        return self.compute_generation_metrics(model, noise_scheduler, test_prompts)