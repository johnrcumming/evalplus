"""
PEFT MoE Decoder Provider for EvalPlus

Supports evaluation of PEFT (Parameter-Efficient Fine-Tuning) models with
Mixture-of-Experts (MoE) architectures, including:
- DyLoRA-MoE
- X-LoRA
- Standard PEFT adapters (LoRA, QLoRA, AdaLoRA)
- W&B artifacts
- Multiple routing strategies

"""



from typing import List, Optional, Dict, Any
import os
import json
import torch
from pathlib import Path

# Enable MPS fallback to CPU for unsupported operations
# This is needed for some PyTorch operations that aren't yet implemented on MPS
# See: https://github.com/pytorch/pytorch/issues/141287
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)

from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


class PeftMoEDecoder(DecoderBase):
    """
    EvalPlus provider for PEFT models with MoE support.
    
    Example usage:
        # DyLoRA-MoE with router
        decoder = PeftMoEDecoder(
            name="./trained_model",
            base_model="google/codegemma-2b",
            routing_strategy="router",
            dataset="humaneval"
        )
        
        # Single expert
        decoder = PeftMoEDecoder(
            name="./trained_model",
            base_model="google/codegemma-2b",
            routing_strategy="single:0",
            dataset="humaneval"
        )
        
        # W&B artifact
        decoder = PeftMoEDecoder(
            name="user/project/model:v0",
            wandb_artifact="user/project/model:v0",
            base_model="google/codegemma-2b",
            dataset="humaneval"
        )
    """
    
    def __init__(
        self,
        name: str,
        dataset: str,
        base_model: Optional[str] = None,
        adapter_path: Optional[str] = None,
        routing_strategy: str = "router",
        wandb_artifact: Optional[str] = None,
        force_base_prompt: bool = False,
        attn_implementation: str = "eager",
        device_map: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize PEFT MoE decoder.
        
        Args:
            name: Model name/path or W&B artifact
            dataset: Dataset name (for EOS tokens)
            base_model: Base model name (e.g., "google/codegemma-2b")
            adapter_path: Explicit adapter path (optional)
            routing_strategy: Expert routing strategy:
                - "router": Use model's internal router (default)
                - "single:<id>": Use specific expert (e.g., "single:0")
                - "best": Analyze prompt, select best expert
                - "ensemble": Generate with all experts, combine
                - "round_robin": Cycle through experts
            wandb_artifact: W&B artifact path (e.g., "user/project/model:v0")
            force_base_prompt: Force base model prompts (not chat)
            attn_implementation: Attention implementation ("eager", "flash_attention_2", "sdpa")
            device_map: Device map for model loading
            **kwargs: Additional arguments for DecoderBase
        """
        super().__init__(name=name, **kwargs)
        
        self.base_model_name = base_model
        self.adapter_path = adapter_path
        self.routing_strategy = routing_strategy
        self.wandb_artifact = wandb_artifact
        self.force_base_prompt = force_base_prompt
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.dataset = dataset
        
        # Device setup - check for CUDA, MPS, or fallback to CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Model format detection
        self.model_format = None
        self.model_path = name
        
        # Load model
        self._load_model()
        
        # Setup tokenizer
        self._setup_tokenizer()
        
        # Setup EOS tokens
        if self.is_direct_completion():
            self.eos += extra_eos_for_direct_completion(dataset)
        else:
            self.eos += ["\n```\n"]
        
        print(f"\n{'='*80}")
        print("PeftMoEDecoder initialized:")
        print(f"  Model format: {self.model_format}")
        print(f"  Base model: {self.base_model_name}")
        print(f"  Routing strategy: {self.routing_strategy}")
        print(f"  Device: {self.device}")
        print(f"  EOS tokens: {self.eos}")
        print(f"{'='*80}\n")
    
    def _load_model(self):
        """Load PEFT MoE model based on detected format."""
        
        # Step 0: Handle adapter_path if provided (takes precedence)
        if self.adapter_path:
            print(f"ℹ️  Using explicit adapter_path: {self.adapter_path}")
            self.model_path = self.adapter_path
        
        # Step 1: Handle W&B artifact if provided
        # Check if wandb_artifact is actually a local path (not a W&B artifact)
        elif self.wandb_artifact:
            # W&B artifacts have format: "entity/project/artifact:version"
            # Local paths don't have this format
            is_local_path = os.path.exists(self.wandb_artifact) or "/" in self.wandb_artifact and not ":" in os.path.basename(self.wandb_artifact)
            
            if is_local_path:
                print(f"ℹ️  Treating wandb_artifact as local path: {self.wandb_artifact}")
                self.model_path = self.wandb_artifact
                self.wandb_artifact = None  # Clear it so we don't try to download
            else:
                print(f"Downloading W&B artifact: {self.wandb_artifact}")
                self.model_path = self._load_from_wandb_artifact(self.wandb_artifact)
                print(f"✓ Artifact downloaded to: {self.model_path}")
        
        # Step 2: Detect model format
        self.model_format = self._detect_model_format(self.model_path)
        print(f"Detected model format: {self.model_format}")
        
        # Step 3: Load base model if needed
        if self.base_model_name is None:
            self.base_model_name = self._infer_base_model(self.model_path)
            print(f"Inferred base model: {self.base_model_name}")
        
        # Step 4: Load model based on format
        if self.model_format == "dylora_moe":
            self.model = self._load_dylora_moe_model()
        elif self.model_format == "peft_adapter":
            self.model = self._load_peft_adapter_model()
        elif self.model_format == "merged":
            self.model = self._load_merged_model()
        else:
            raise ValueError(f"Unknown model format: {self.model_format}")
        
        # Step 5: Move to device if needed
        if self.device_map is None:
            self.model = self.model.to(self.device)
        
        print(f"✓ Model loaded successfully")

    def _detect_model_format(self, model_path: str) -> str:
        """
        Detect PEFT model format.
        
        Returns:
            "dylora_moe": DyLoRA-MoE with router state
            "peft_adapter": Standard PEFT adapter
            "merged": Merged PEFT model
        """
        model_path = Path(model_path)
        
        # Check for DyLoRA-MoE indicators
        # 1. Check for dylo_moe_state directory (legacy)
        if (model_path.parent / "dylo_moe_state").exists():
            return "dylora_moe"
        
        # 2. Check config.json for DyLoRA-MoE marker
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    if config.get("model_type") == "dylora-moe":
                        return "dylora_moe"
                    if config.get("_dylora_original_type") == "dylora-moe":
                        return "dylora_moe"
            except Exception as e:
                print(f"⚠️  Warning: Could not parse config.json: {e}")
        
        # 3. Check for PEFT adapter
        if (model_path / "adapter_config.json").exists():
            return "peft_adapter"
        
        # 4. Default to merged if has model weights
        if (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists():
            return "merged"
        
        raise ValueError(f"Cannot determine model format for: {model_path}")

    def _infer_base_model(self, model_path: str) -> str:
        """
        Infer base model name from config.
        
        Returns:
            Base model name (e.g., "google/codegemma-2b")
        """
        config_path = Path(model_path) / "config.json"
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path")
                    if base_model:
                        return base_model
            except Exception as e:
                print(f"⚠️  Warning: Could not parse config.json: {e}")
        
        # Check adapter_config.json for PEFT adapters
        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path) as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path")
                    if base_model:
                        return base_model
            except Exception as e:
                print(f"⚠️  Warning: Could not parse adapter_config.json: {e}")
        
        # Default fallback
        default = "google/codegemma-2b"
        print(f"⚠️  Could not infer base model, using default: {default}")
        return default

    def _load_from_wandb_artifact(self, artifact_path: str) -> str:
        """
        Download W&B artifact and return local path.
        
        Args:
            artifact_path: Format "entity/project/artifact:version"
        
        Returns:
            Local path to downloaded artifact
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for artifact loading. "
                "Install with: pip install wandb"
            )
        
        # Initialize W&B run for artifact download
        # Use online mode to enable artifact downloads, but disable syncing to avoid creating runs
        import os
        os.environ['WANDB_SILENT'] = 'true'  # Suppress W&B output
        
        with wandb.init(mode="online", anonymous="allow", settings=wandb.Settings(silent=True)) as run:
            artifact = run.use_artifact(artifact_path, type='model')
            artifact_dir = artifact.download()
        
        # Look for model subdirectory
        artifact_path = Path(artifact_dir)
        
        # Check for best_model subdirectory (common in training artifacts)
        best_model_path = artifact_path / "best_model"
        if best_model_path.exists():
            return str(best_model_path)
        
        return str(artifact_path)

    def _load_dylora_moe_model(self):
        """Load DyLoRA-MoE model with separated PEFT adapters."""
        print("Loading DyLoRA-MoE model...")
        
        # Try importing DyLoRA-MoE
        try:
            from dylo_moe.model import DyLoRA_MoE
            from dylo_moe.expert import ExpertManager
            from dylo_moe.utils import load_lora_experts
        except ImportError:
            raise ImportError(
                "DyLoRA-MoE modules not found. "
                "Ensure dylo_moe package is available in PYTHONPATH."
            )
        
        # Load config to get parameters
        config_path = Path(self.model_path) / "config.json"
        
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                num_experts = config.get("_dylora_num_experts", config.get("num_experts", 4))
                lora_r = config.get("_dylora_lora_r", config.get("lora_r", 16))
                lora_alpha = config.get("_dylora_lora_alpha", config.get("lora_alpha", 32))
                lora_dropout = config.get("_dylora_lora_dropout", config.get("lora_dropout", 0.05))
        else:
            # Defaults
            num_experts = 4
            lora_r = 16
            lora_alpha = 32
            lora_dropout = 0.05
            print(f"⚠️  No config.json found, using defaults: num_experts={num_experts}, lora_r={lora_r}")
        
        # Initialize DyLoRA_MoE model
        print(f"Initializing DyLoRA-MoE: {num_experts} experts, r={lora_r}, alpha={lora_alpha}")
        model = DyLoRA_MoE(
            model_name=self.base_model_name,
            num_experts=num_experts,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            token=os.environ.get("HF_TOKEN"),
        )
        
        # Check for PEFT adapters directory (new format with separated experts)
        peft_adapters_dir = Path(self.model_path) / "peft_adapters"
        if peft_adapters_dir.exists():
            print(f"✓ Found PEFT adapters directory: {peft_adapters_dir}")
            print(f"  Loading {num_experts} expert adapters from separate files...")
            
            try:
                load_lora_experts(model, str(peft_adapters_dir))
                print(f"✓ Loaded all expert adapters from {peft_adapters_dir}")
            except Exception as e:
                print(f"⚠️  Failed to load expert adapters: {e}")
                print("  Falling back to base model weights only")
        else:
            # Fallback: Try loading from merged weights file (legacy format)
            print("⚠️  No peft_adapters directory found, trying legacy merged weights...")
            
            weights_file = None
            safetensors_path = Path(self.model_path) / "model.safetensors"
            pytorch_path = Path(self.model_path) / "pytorch_model.bin"
            
            if safetensors_path.exists():
                weights_file = safetensors_path
                print(f"Loading merged weights from: {weights_file.name}")
                from safetensors.torch import load_file
                state_dict = load_file(str(weights_file))
            elif pytorch_path.exists():
                weights_file = pytorch_path
                print(f"Loading merged weights from: {weights_file.name}")
                state_dict = torch.load(str(weights_file), map_location="cpu")
            else:
                print("⚠️  No weight files found, using base model weights only")
                return model
            
            # Load merged state dict (legacy)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            
            print(f"✓ Loaded merged weights from: {weights_file.name}")
            print("⚠️  WARNING: Using merged weights - MoE routing may not work properly")
        
        # Load router state if available
        router_state_path = Path(self.model_path) / "dylo_moe_state" / "router.pt"
        if router_state_path.exists():
            print(f"Loading router state from: {router_state_path}")
            try:
                # Load router with proper device mapping
                from dylo_moe.router import DynamicHybridRouter
                model.router = DynamicHybridRouter.load(str(router_state_path), device=self.device)
                print(f"✓ Loaded router state")
            except Exception as e:
                print(f"⚠️  Failed to load router state: {e}")
        else:
            print("⚠️  No router state found, using initialized router")
        
        return model

    def _load_peft_adapter_model(self):
        """Load standard PEFT adapter model."""
        print("Loading PEFT adapter model...")
        
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError:
            raise ImportError(
                "PEFT library required. Install with: pip install peft"
            )
        
        # Load using PEFT's auto loader
        model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        
        print(f"✓ Loaded PEFT adapter model")
        return model

    def _load_merged_model(self):
        """Load merged PEFT model (adapters already merged into base)."""
        print("Loading merged model...")
        
        # Load as standard HuggingFace model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self.attn_implementation,
        )
        
        print(f"✓ Loaded merged model")
        return model

    def _setup_tokenizer(self):
        """Setup tokenizer for the model."""
        
        # Try loading tokenizer from model path first
        tokenizer_path = Path(self.model_path)
        
        if (tokenizer_path / "tokenizer.json").exists() or (tokenizer_path / "tokenizer_config.json").exists():
            print(f"Loading tokenizer from model path: {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                trust_remote_code=self.trust_remote_code
            )
        else:
            # Load from base model
            print(f"Loading tokenizer from base model: {self.base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                token=os.environ.get("HF_TOKEN"),
                trust_remote_code=self.trust_remote_code
            )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.skip_special_tokens = True
        
        print(f"✓ Tokenizer setup complete")

    def is_direct_completion(self) -> bool:
        """Check if using direct completion (no chat template)."""
        return self.force_base_prompt or self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        """
        Generate code completions using PEFT MoE model.
        
        Args:
            prompt: Input prompt
            do_sample: Use sampling (vs greedy)
            num_samples: Number of completions to generate
        
        Returns:
            List of generated code strings
        """
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1
        
        # Prepare prompt (with chat template if applicable)
        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
            )
        )
        
        # Tokenize
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.device_map is None:
            input_tokens = input_tokens.to(self.device)
        
        # Setup generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": min(self.batch_size, num_samples),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        if do_sample:
            gen_kwargs["top_p"] = 0.95
            gen_kwargs["temperature"] = self.temperature
        
        # Apply routing strategy
        self._apply_routing_strategy()
        
        # Generate
        outputs = self.model.generate(input_tokens, **gen_kwargs)
        
        # Decode
        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1):],
            skip_special_tokens=self.skip_special_tokens,
        )
        
        # Post-process: remove EOS tokens
        processed_outputs = []
        for output in gen_strs:
            min_index = len(output)
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            processed_outputs.append(output[:min_index].replace("\t", "    "))
        
        return processed_outputs

    def _apply_routing_strategy(self):
        """Apply routing strategy to model before generation."""
        
        # Check if model has expert management capabilities
        has_expert_manager = (
            hasattr(self.model, 'expert_manager') and 
            hasattr(self.model, 'router')
        )
        
        if not has_expert_manager:
            # No expert management - skip routing
            if self.routing_strategy != "router":
                print(f"⚠️  Model doesn't support routing, ignoring strategy: {self.routing_strategy}")
            return
        
        # Parse routing strategy
        if self.routing_strategy == "router":
            # Use model's internal router - activate all experts for MoE
            if self.model.expert_manager.num_experts > 1:
                self.model.expert_manager.activate_all_experts()
                # print(f"Using router with {self.model.expert_manager.num_experts} experts")
        
        elif self.routing_strategy.startswith("single:"):
            # Single expert mode
            expert_id = int(self.routing_strategy.split(":")[1])
            if expert_id >= self.model.expert_manager.num_experts:
                raise ValueError(
                    f"Expert {expert_id} not found. "
                    f"Model has {self.model.expert_manager.num_experts} experts."
                )
            self.model.expert_manager.set_active_expert(expert_id)
            # print(f"Using single expert: {expert_id}")
        
        elif self.routing_strategy == "best":
            # Analyze prompt and select best expert
            # For now, default to expert 0
            # TODO: Implement prompt analysis in future enhancement
            self.model.expert_manager.set_active_expert(0)
            # print(f"Using 'best' strategy (selecting expert 0 for now)")
        
        elif self.routing_strategy == "ensemble":
            # Ensemble mode - activate all experts
            self.model.expert_manager.activate_all_experts()
            # print(f"Using ensemble with {self.model.expert_manager.num_experts} experts")
        
        elif self.routing_strategy == "round_robin":
            # Round robin - for now, just use router
            # TODO: Implement round robin in future enhancement
            if self.model.expert_manager.num_experts > 1:
                self.model.expert_manager.activate_all_experts()
            # print(f"Using round robin (defaulting to router for now)")
        
        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")
