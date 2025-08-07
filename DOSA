# dosa.py
# A reference implementation of Dynamic Orthogonalized Subspace Adaptation (DOSA)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
from typing import List, Dict, Any

# --- Core DOSA Components ---

class StructuredKernel(nn.Module):
    """Abstract base class for structured kernels used in DOSA."""
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def get_S_theta(self) -> torch.Tensor:
        raise NotImplementedError

class BlockDiagonalKernel(StructuredKernel):
    """
    A block-diagonal structured kernel. This is a parameter-efficient way
    to represent the update matrix S_theta by using a series of smaller,
    independent linear layers.
    """
    def __init__(self, features: int, num_blocks: int):
        """
        Args:
            features (int): The total number of input/output features.
            num_blocks (int): The number of diagonal blocks. Must divide features.
        """
        super().__init__()
        if features % num_blocks != 0:
            raise ValueError("features must be divisible by num_blocks")
        self.features, self.num_blocks = features, num_blocks
        self.block_size = features // num_blocks
        
        # Create a list of small linear layers (the blocks)
        self.blocks = nn.ModuleList([
            nn.Linear(self.block_size, self.block_size, bias=False) for _ in range(num_blocks)
        ])
        # Initialize the kernel weights to zero
        for block in self.blocks:
            nn.init.zeros_(block.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block-diagonal transformation."""
        batch_size, seq_len, _ = x.shape
        # Reshape input to isolate features for each block
        # (batch*seq_len, num_blocks, block_size)
        x_reshaped = x.view(batch_size * seq_len, self.num_blocks, self.block_size)
        
        # Apply each block to its corresponding chunk of features
        output_chunks = [self.blocks[i](x_reshaped[:, i, :]) for i in range(self.num_blocks)]
        
        # Concatenate the results and restore original shape
        output = torch.cat(output_chunks, dim=1)
        return output.view(batch_size, seq_len, self.features)

    @torch.no_grad()
    def get_S_theta(self) -> torch.Tensor:
        """
        Constructs the full S_theta matrix from the individual blocks for merging.
        Returns:
            torch.Tensor: The full (features x features) sparse kernel matrix.
        """
        S_theta = torch.zeros(self.features, self.features, device=self.blocks[0].weight.device)
        for i in range(self.num_blocks):
            start, end = i * self.block_size, (i + 1) * self.block_size
            S_theta[start:end, start:end] = self.blocks[i].weight.data
        return S_theta

class DOSALayer(nn.Module):
    """
    A linear layer adapted with Dynamic Orthogonalized Subspace Adaptation (DOSA).
    This layer wraps a frozen linear layer and applies a small, trainable update
    that is projected into the null space of the original layer's activations.
    """
    def __init__(self, in_features: int, out_features: int, kernel_type: str = 'block_diagonal', kernel_config: Dict = None, alpha_init: float = 0.01):
        super().__init__()
        if in_features != out_features:
            raise ValueError(f"DOSALayer requires square matrices: in_features == out_features.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.merged = False

        # 1. The original, frozen linear layer
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # 2. The small, trainable structured kernel (S_theta)
        if kernel_type == 'block_diagonal':
            kernel_config = kernel_config if kernel_config is not None else {'num_blocks': 4}
            self.kernel = BlockDiagonalKernel(in_features, **kernel_config)
        else:
            raise NotImplementedError(f"Kernel type '{kernel_type}' is not implemented.")

        # 3. The trainable scaling factor
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # 4. The fixed projection matrix (P), initialized to identity
        self.register_buffer('proj_matrix', torch.eye(in_features))
        self.proj_matrix_loaded = False

    def set_projection_matrix(self, proj_matrix: torch.Tensor):
        """
        Loads the pre-computed projection matrix.
        Args:
            proj_matrix (torch.Tensor): The (features x features) projection matrix.
        """
        if proj_matrix.shape != (self.in_features, self.in_features):
            raise ValueError(f"Projection matrix must have shape ({self.in_features}, {self.in_features})")
        self.proj_matrix = proj_matrix.to(self.linear.weight.device)
        self.proj_matrix_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the output as: Wx + alpha * P * S_theta(x)
        """
        if self.merged or not self.proj_matrix_loaded:
            return self.linear(x)

        base_output = self.linear(x)
        kernel_output = self.kernel(x)
        
        # Project the kernel output into the null space
        projected_update = F.linear(kernel_output, self.proj_matrix)
        
        return base_output + self.alpha * projected_update

    @torch.no_grad()
    def merge(self):
        """
        Merges the learned update into the weights of the linear layer.
        This eliminates computational overhead during inference.
        """
        if self.merged or not self.proj_matrix_loaded:
            return
        
        # Calculate the total weight update: Delta_W^T = alpha * S_theta^T @ P
        # Note: P is symmetric (P = P^T), so P can be on the right.
        S_theta_T = self.kernel.get_S_theta().T
        delta_W_T = (S_theta_T @ self.proj_matrix) * self.alpha
        
        # Apply the update
        self.linear.weight.data += delta_W_T
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """Reverts the merge operation."""
        if not self.merged or not self.proj_matrix_loaded:
            return
            
        S_theta_T = self.kernel.get_S_theta().T
        delta_W_T = (S_theta_T @ self.proj_matrix) * self.alpha
        
        self.linear.weight.data -= delta_W_T
        self.merged = False


# --- Utility Functions for Model Integration ---

@torch.no_grad()
def compute_projection_matrix(
    model: nn.Module, 
    layer_name: str, 
    data_loader: DataLoader, 
    k_dim: int, 
    device: str,
    model_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Computes the projection matrix for a layer by finding the null space
    of its input activations using Singular Value Decomposition (SVD).

    Args:
        model (nn.Module): The model containing the layer.
        layer_name (str): The fully-qualified name of the target layer.
        data_loader (DataLoader): DataLoader providing calibration data.
        k_dim (int): The starting dimension of the null space. All singular vectors
                     from k_dim onwards will be used.
        device (str): The device to perform computations on ('cuda' or 'cpu').
        model_dtype (torch.dtype): The data type for the final matrix.

    Returns:
        torch.Tensor: The computed (features x features) projection matrix.
    """
    print(f"Computing projection matrix for layer: {layer_name}")
    target_layer = model.get_submodule(layer_name)
    activations = []
    
    hook = target_layer.register_forward_hook(
        lambda module, input, output: activations.append(
            input[0].detach().float().cpu().view(-1, input[0].shape[-1])
        )
    )
    
    model.eval()
    for batch in tqdm(data_loader, desc=f"Collecting activations for {layer_name}"):
        inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        model(**inputs, use_cache=False)
    
    hook.remove()
    
    if not activations:
        raise RuntimeError(f"No activations collected for layer {layer_name}.")

    H_in = torch.cat(activations, dim=0).float()
    print(f"Activation matrix shape: {H_in.shape}")
    
    print("Computing SVD on the covariance matrix...")
    # U corresponds to the eigenvectors of H_in.T @ H_in
    U, _, _ = torch.linalg.svd(H_in.T @ H_in)
    
    # The null space is spanned by the eigenvectors with the smallest eigenvalues
    V_null = U[:, k_dim:]
    
    # The projection matrix P = V_null @ V_null.T
    proj_matrix = V_null @ V_null.T
    
    print(f"Projection matrix computed. Shape: {proj_matrix.shape}")
    
    # Clean up memory
    del H_in, U, V_null, activations
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return proj_matrix.to(device=device, dtype=model_dtype)

def apply_dosa_to_model(model: nn.Module, dosa_config: Dict) -> List[str]:
    """
    Finds target linear layers in a model and replaces them with DOSALayers.

    Args:
        model (nn.Module): The model to modify.
        dosa_config (Dict): Configuration dictionary specifying target_modules,
                            kernel_type, kernel_config, etc.

    Returns:
        List[str]: A list of names of the layers that were replaced.
    """
    layers_to_replace = []
    target_modules = dosa_config.get('target_modules', [])
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            if module.in_features == module.out_features:
                layers_to_replace.append(name)
    
    for name in layers_to_replace:
        original_layer = model.get_submodule(name)
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent_module = model.get_submodule(parent_name) if parent_name else model
        
        dosa_layer = DOSALayer(
            original_layer.in_features,
            original_layer.out_features,
            kernel_type=dosa_config.get('kernel_type', 'block_diagonal'),
            kernel_config=dosa_config.get('kernel_config'),
            alpha_init=dosa_config.get('alpha_init', 0.01)
        )
        dosa_layer.linear.weight.data = original_layer.weight.data.clone()
        if original_layer.bias is not None:
             # DOSA implementation does not handle bias updates, so we discard it.
             # The original bias is in the frozen part, which is often sufficient.
             dosa_layer.linear.bias = None 

        setattr(parent_module, child_name, dosa_layer)
        print(f"Replaced '{name}' with DOSALayer.")
        
    return layers_to_replace

def get_dosa_trainable_parameters(model: nn.Module) -> None:
    """
    Freezes all model parameters except for the trainable components of DOSALayers
    (the kernel and alpha) and prints a summary.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    
    for name, param in model.named_parameters():
        # Only 'kernel' and 'alpha' parameters within DOSALayers are trainable
        if 'kernel' in name or 'alpha' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            
    print("-" * 50)
    print(f"Total params:         {total_params/1e6:.2f} M")
    print(f"Trainable DOSA params: {trainable_params/1e3:.2f} K")
    print(f"Trainable percentage:  {100 * trainable_params / total_params:.4f}%")
    print("-" * 50)

# --- Example Usage ---
if __name__ == '__main__':
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
    
    # 1. Configuration
    MODEL_NAME = "google/flan-t5-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DOSA_CONFIG = {
        'target_modules': ['q', 'v'],  # Target query and value projections in attention
        'kernel_type': 'block_diagonal',
        'kernel_config': {'num_blocks': 8}, 
        'alpha_init': 0.01,
        'k': 640 # Null space dimension (hyperparameter)
    }

    # 2. Load a pre-trained model
    print(f"Loading model '{MODEL_NAME}'...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_cache=False).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. Apply DOSA to the model
    print("\nApplying DOSA layers...")
    dosa_layer_names = apply_dosa_to_model(model, DOSA_CONFIG)
    
    # 4. Calibrate: Compute projection matrices
    # This requires a small, representative sample of your task data.
    # Here we create dummy data for demonstration.
    print("\nPreparing for calibration...")
    dummy_texts = ["translate English to French: Hello world", "summarize: The quick brown fox jumps over the lazy dog."]
    dummy_inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True)
    dummy_dataset = [{'input_ids': dummy_inputs['input_ids'][i], 'attention_mask': dummy_inputs['attention_mask'][i], 'labels': dummy_inputs['input_ids'][i]} for i in range(len(dummy_texts))]
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    calibration_loader = DataLoader(dummy_dataset, batch_size=2, collate_fn=collator)

    for name in dosa_layer_names:
        proj_matrix = compute_projection_matrix(model, name, calibration_loader, DOSA_CONFIG['k'], DEVICE)
        model.get_submodule(name).set_projection_matrix(proj_matrix)

    # 5. Set up trainable parameters for fine-tuning
    print("\nSetting up trainable parameters...")
    get_dosa_trainable_parameters(model)
    
    print("\nModel is now ready for fine-tuning with a standard training loop.")
    print("Only the DOSA parameters (kernels and alphas) will be updated.")

    # (Your fine-tuning loop using Hugging Face Trainer or custom PyTorch loop would go here)
    # trainer.train()

    # 6. After training, merge weights for efficient inference
    print("\nMerging trained DOSA weights for deployment...")
    for module in model.modules():
        if isinstance(module, DOSALayer):
            module.merge()
    print("Merge complete. The model now behaves like a standard model with no overhead.")
