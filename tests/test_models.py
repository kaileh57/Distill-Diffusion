import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.diffusion_transformer import DiffusionTransformer, DiffusionTransformerConfig, TimestepEmbedding
from models.noise_scheduler import MaskedDiffusionScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@pytest.fixture
def base_model(tokenizer):
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    return model

@pytest.fixture
def diffusion_config(tokenizer):
    return DiffusionTransformerConfig(
        base_model_name='gpt2',
        hidden_size=768,
        num_timesteps=100,
        mask_token_id=tokenizer.mask_token_id,
        use_bidirectional_attention=True,
        freeze_base_model=True
    )

@pytest.fixture
def diffusion_model(base_model, diffusion_config, device):
    model = DiffusionTransformer(base_model, diffusion_config)
    model.to(device)
    return model

@pytest.fixture
def noise_scheduler(tokenizer):
    return MaskedDiffusionScheduler(
        num_timesteps=100,
        schedule_type="cosine",
        mask_token_id=tokenizer.mask_token_id
    )

def test_timestep_embedding():
    embed = TimestepEmbedding(128)
    timesteps = torch.tensor([0, 10, 100, 999])
    
    embeddings = embed(timesteps)
    assert embeddings.shape == (4, 128)
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()

def test_noise_scheduler_creation(noise_scheduler):
    assert noise_scheduler.num_timesteps == 100
    assert noise_scheduler.schedule_type == "cosine"
    assert noise_scheduler.mask_token_id is not None

def test_noise_scheduler_noise_addition(noise_scheduler, device):
    # Test noise addition
    input_ids = torch.randint(0, 50000, (2, 10), device=device)
    timesteps = torch.tensor([10, 50], device=device)
    
    noisy_ids, mask = noise_scheduler.add_noise(input_ids, timesteps)
    
    assert noisy_ids.shape == input_ids.shape
    assert mask.shape == input_ids.shape
    assert mask.dtype == torch.bool
    assert (noisy_ids[mask] == noise_scheduler.mask_token_id).all()

def test_noise_scheduler_noise_levels(noise_scheduler):
    # Test noise levels are reasonable
    for t in [0, 25, 50, 75, 99]:
        noise_level = noise_scheduler.get_noise_level(t)
        assert 0 <= noise_level <= 1
        assert not torch.isnan(torch.tensor(noise_level))

def test_diffusion_model_creation(diffusion_model):
    assert diffusion_model is not None
    assert hasattr(diffusion_model, 'base_model')
    assert hasattr(diffusion_model, 'time_embed')
    assert hasattr(diffusion_model, 'time_mlp')
    assert hasattr(diffusion_model, 'denoising_head')

def test_diffusion_model_forward_pass(diffusion_model, tokenizer, device):
    # Test forward pass
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=20)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    timesteps = torch.tensor([50], device=device)
    
    diffusion_model.eval()
    with torch.no_grad():
        outputs = diffusion_model(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask
        )
    
    assert isinstance(outputs, dict)
    assert 'logits' in outputs
    assert 'hidden_states' in outputs
    assert 'timesteps' in outputs
    
    logits = outputs['logits']
    assert logits.shape == (1, input_ids.shape[1], len(tokenizer))
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

def test_bidirectional_attention_difference(diffusion_model, tokenizer, device):
    # Test that bidirectional attention produces different results than causal
    text = "The cat sat on the [MASK] mat."
    inputs = tokenizer(text.replace('[MASK]', tokenizer.mask_token), 
                      return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    timesteps = torch.tensor([50], device=device)
    
    diffusion_model.eval()
    
    # Test with bidirectional attention
    diffusion_model.config.use_bidirectional_attention = True
    with torch.no_grad():
        outputs_bi = diffusion_model(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask
        )
    
    # Test with causal attention
    diffusion_model.config.use_bidirectional_attention = False
    with torch.no_grad():
        outputs_causal = diffusion_model(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask
        )
    
    # Results should be different
    assert not torch.allclose(outputs_bi['logits'], outputs_causal['logits'], atol=1e-4)
    
    # Reset to bidirectional
    diffusion_model.config.use_bidirectional_attention = True

def test_model_freezing(diffusion_model):
    # Test that base model is frozen when specified
    base_requires_grad = any(p.requires_grad for p in diffusion_model.base_model.parameters())
    diffusion_requires_grad = any(p.requires_grad for p in diffusion_model.time_embed.parameters())
    head_requires_grad = any(p.requires_grad for p in diffusion_model.denoising_head.parameters())
    
    # Base model should be frozen
    assert not base_requires_grad
    # But diffusion components should be trainable
    assert diffusion_requires_grad
    assert head_requires_grad
    
    # Test unfreezing
    diffusion_model.unfreeze_base_model()
    base_requires_grad_after = any(p.requires_grad for p in diffusion_model.base_model.parameters())
    assert base_requires_grad_after

def test_token_embedding_resize(diffusion_model, tokenizer):
    original_vocab_size = diffusion_model.denoising_head.out_features
    
    # Add new tokens
    new_tokens = ['<special1>', '<special2>']
    tokenizer.add_tokens(new_tokens)
    
    # Resize embeddings
    diffusion_model.resize_token_embeddings(len(tokenizer))
    
    new_vocab_size = diffusion_model.denoising_head.out_features
    assert new_vocab_size == len(tokenizer)
    assert new_vocab_size == original_vocab_size + len(new_tokens)

def test_different_batch_sizes(diffusion_model, tokenizer, device):
    # Test with different batch sizes
    texts = [
        "Short text.",
        "This is a longer text with more tokens to test batching.",
        "Medium length text for testing."
    ]
    
    for batch_size in [1, 2, 3]:
        batch_texts = texts[:batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        timesteps = torch.tensor([25] * batch_size, device=device)
        
        diffusion_model.eval()
        with torch.no_grad():
            outputs = diffusion_model(
                input_ids=input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask
            )
        
        assert outputs['logits'].shape[0] == batch_size
        assert not torch.isnan(outputs['logits']).any()

def test_different_timesteps(diffusion_model, tokenizer, device):
    # Test with different timesteps
    text = "Test text for timestep variation."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    timesteps_list = [0, 25, 50, 75, 99]
    
    diffusion_model.eval()
    previous_logits = None
    
    for t in timesteps_list:
        timesteps = torch.tensor([t], device=device)
        
        with torch.no_grad():
            outputs = diffusion_model(
                input_ids=input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask
            )
        
        logits = outputs['logits']
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        
        # Different timesteps should produce different results
        if previous_logits is not None:
            assert not torch.allclose(logits, previous_logits, atol=1e-4)
        
        previous_logits = logits

if __name__ == "__main__":
    # Run tests directly
    print("Running basic tests...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test timestep embedding
    print("Testing timestep embedding...")
    test_timestep_embedding()
    print("✓ Timestep embedding test passed")
    
    # Test noise scheduler
    print("Testing noise scheduler...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    noise_scheduler = MaskedDiffusionScheduler(
        num_timesteps=100,
        schedule_type="cosine",
        mask_token_id=tokenizer.mask_token_id
    )
    
    test_noise_scheduler_creation(noise_scheduler)
    test_noise_scheduler_noise_addition(noise_scheduler, device)
    test_noise_scheduler_noise_levels(noise_scheduler)
    print("✓ Noise scheduler tests passed")
    
    print("\nAll basic tests passed!")
    print("Run 'pytest tests/test_models.py' for full test suite with fixtures.")