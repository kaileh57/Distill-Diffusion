"""
Diffusion transformer architecture adapted from autoregressive models.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from einops import rearrange
import math
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings for diffusion."""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: Tensor of shape (batch_size,) containing timestep values
            
        Returns:
            Tensor of shape (batch_size, dim) containing embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding

class DiffusionTransformerConfig(PretrainedConfig):
    """Configuration for DiffusionTransformer."""
    
    def __init__(
        self,
        base_model_name: str = None,
        hidden_size: int = 768,
        num_timesteps: int = 1000,
        mask_token_id: int = None,
        time_embedding_dim: int = None,
        use_bidirectional_attention: bool = True,
        freeze_base_model: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.hidden_size = hidden_size
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        self.time_embedding_dim = time_embedding_dim or hidden_size
        self.use_bidirectional_attention = use_bidirectional_attention
        self.freeze_base_model = freeze_base_model

class DiffusionTransformer(PreTrainedModel):
    """
    Wrapper to convert autoregressive transformer to diffusion model.
    """
    config_class = DiffusionTransformerConfig
    
    def __init__(self, base_model, config):
        super().__init__(config)
        self.base_model = base_model
        self.config = config
        
        # Get base model config
        self.base_config = base_model.config
        
        # Freeze base model initially if specified
        if config.freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Add diffusion-specific components
        self.time_embed = TimestepEmbedding(config.time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(0.1)
        )
        
        # Add denoising head
        self.denoising_head = nn.Linear(
            config.hidden_size, 
            self.base_config.vocab_size,
            bias=False
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize additional weights."""
        # Initialize time MLP
        for module in self.time_mlp:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        # Initialize denoising head
        torch.nn.init.normal_(self.denoising_head.weight, std=0.02)
        
    def unfreeze_base_model(self):
        """Unfreeze base model parameters for fine-tuning."""
        for param in self.base_model.parameters():
            param.requires_grad = True
            
    def freeze_base_model(self):
        """Freeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def get_input_embeddings(self):
        """Get input embeddings from base model."""
        if hasattr(self.base_model, 'transformer'):
            return self.base_model.transformer.wte
        elif hasattr(self.base_model, 'model'):
            return self.base_model.model.embed_tokens
        else:
            raise ValueError("Cannot find input embeddings in base model")
    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings."""
        self.base_model.resize_token_embeddings(new_num_tokens)
        # Update denoising head
        old_head = self.denoising_head
        self.denoising_head = nn.Linear(
            self.config.hidden_size,
            new_num_tokens,
            bias=False
        ).to(old_head.weight.device)
        
        # Copy old weights
        min_tokens = min(old_head.out_features, new_num_tokens)
        with torch.no_grad():
            self.denoising_head.weight[:min_tokens] = old_head.weight[:min_tokens]
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        timesteps: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        Forward pass through diffusion transformer.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            timesteps: Timestep values of shape (batch_size,)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            return_dict: Whether to return a dict or tuple
            
        Returns:
            Logits for denoising prediction
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get time embeddings
        time_emb = self.time_embed(timesteps)  # (batch_size, time_embed_dim)
        time_emb = self.time_mlp(time_emb)      # (batch_size, hidden_size)
        
        # Get base model embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)  # (batch_size, seq_len, hidden_size)
        
        # Add time embeddings to all positions
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_size)
        inputs_embeds = inputs_embeds + time_emb
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Modify attention mask for bidirectional attention if needed
        if self.config.use_bidirectional_attention:
            # Create bidirectional attention mask
            causal_mask = None
        else:
            # Keep causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Forward through transformer
        try:
            outputs = self._forward_transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                causal_mask=causal_mask
            )
        except Exception as e:
            logger.error(f"Error in transformer forward pass: {e}")
            raise
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        
        # Apply denoising head
        logits = self.denoising_head(hidden_states)
        
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': hidden_states,
                'timesteps': timesteps
            }
        else:
            return logits
    
    def _forward_transformer(self, inputs_embeds, attention_mask, causal_mask=None):
        """Forward pass through transformer layers."""
        # This is model-specific and depends on the architecture
        
        if hasattr(self.base_model, 'transformer'):
            # GPT-style models (GPT-2, GPT-Neo, etc.)
            return self._forward_gpt_style(inputs_embeds, attention_mask, causal_mask)
        elif hasattr(self.base_model, 'model'):
            # LLaMA-style models
            return self._forward_llama_style(inputs_embeds, attention_mask, causal_mask)
        else:
            raise ValueError("Unsupported model architecture")
    
    def _forward_gpt_style(self, inputs_embeds, attention_mask, causal_mask):
        """Forward pass for GPT-style models."""
        if self.config.use_bidirectional_attention:
            # Override causal mask to enable bidirectional attention
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device
            
            # Create full attention mask (no causal masking)
            full_attention_mask = torch.ones(
                batch_size, seq_len, seq_len, device=device
            )
            
            # Apply padding mask if provided
            if attention_mask is not None:
                # Convert 2D mask to 4D for attention
                extended_attention_mask = attention_mask[:, None, None, :]
                extended_attention_mask = extended_attention_mask.expand(
                    batch_size, 1, seq_len, seq_len
                )
                # Set masked positions to 0
                full_attention_mask = full_attention_mask * extended_attention_mask
            
            # Convert to attention mask format (0 for attend, -inf for mask)
            attention_mask_4d = torch.where(
                full_attention_mask == 0,
                torch.tensor(float('-inf'), device=device),
                torch.tensor(0.0, device=device)
            )
            
            # Store original attention implementation
            original_attn_forward = {}
            
            def create_bidirectional_forward(original_forward):
                def bidirectional_forward(hidden_states, attention_mask=None, **kwargs):
                    # Use our custom attention mask
                    return original_forward(hidden_states, attention_mask=attention_mask_4d, **kwargs)
                return bidirectional_forward
            
            # Temporarily replace attention forward methods
            for i, layer in enumerate(self.base_model.transformer.h):
                if hasattr(layer, 'attn'):
                    original_attn_forward[i] = layer.attn.forward
                    layer.attn.forward = create_bidirectional_forward(original_attn_forward[i])
            
            try:
                outputs = self.base_model.transformer(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True
                )
            finally:
                # Restore original attention methods
                for i, layer in enumerate(self.base_model.transformer.h):
                    if i in original_attn_forward:
                        layer.attn.forward = original_attn_forward[i]
        else:
            outputs = self.base_model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )
        
        return outputs
    
    def _forward_llama_style(self, inputs_embeds, attention_mask, causal_mask):
        """Forward pass for LLaMA-style models."""
        # For LLaMA-style models
        outputs = self.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        
        return outputs
    
    def generate_sample(
        self,
        noise_scheduler,
        prompt_ids: Optional[torch.Tensor] = None,
        max_length: int = 512,
        num_inference_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        device: str = "cuda"
    ):
        """
        Generate samples using diffusion sampling.
        
        Args:
            noise_scheduler: Noise scheduler for sampling
            prompt_ids: Optional prompt token IDs
            max_length: Maximum sequence length
            num_inference_steps: Number of denoising steps
            temperature: Sampling temperature
            top_k: Top-k sampling
            device: Device to use
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        batch_size = 1 if prompt_ids is None else prompt_ids.shape[0]
        
        # Initialize with masked tokens
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            seq_len = max(max_length, prompt_len)
            input_ids = torch.full((batch_size, seq_len), self.config.mask_token_id, device=device)
            input_ids[:, :prompt_len] = prompt_ids
        else:
            input_ids = torch.full((batch_size, max_length), self.config.mask_token_id, device=device)
        
        # Sampling loop
        timesteps = torch.linspace(
            noise_scheduler.num_timesteps - 1, 
            0, 
            num_inference_steps, 
            device=device
        ).long()
        
        with torch.no_grad():
            for t in timesteps:
                # Predict noise
                t_batch = t.expand(batch_size)
                outputs = self(input_ids, t_batch)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from logits
                probs = torch.softmax(logits, dim=-1)
                
                # Only update masked positions
                mask = (input_ids == self.config.mask_token_id)
                if mask.any():
                    # Sample for masked positions
                    masked_probs = probs[mask]
                    sampled_ids = torch.multinomial(masked_probs, num_samples=1).squeeze(-1)
                    input_ids[mask] = sampled_ids
        
        return input_ids

    def save_pretrained(self, save_directory):
        """Save the model."""
        super().save_pretrained(save_directory)
        
        # Save base model separately
        base_model_path = f"{save_directory}/base_model"
        self.base_model.save_pretrained(base_model_path)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load the model."""
        # Load config
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        
        # Load base model
        from transformers import AutoModelForCausalLM
        base_model_path = f"{pretrained_model_name_or_path}/base_model"
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
        
        # Create diffusion model
        model = cls(base_model, config)
        
        # Load additional weights
        state_dict = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        
        return model