"""Configuration for In-Run Data Shapley computation."""

import argparse
import os
import sys
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import build_result_dir


# Directory configurations
RESULTS_DIR = './results/inrun_shapley'


def parse_arguments():
    """Parse command line arguments for In-Run Data Shapley."""
    parser = argparse.ArgumentParser(description='In-Run Data Shapley computation.')
    
    # Method parameters
    parser.add_argument('--method', type=str, default='InRunShapley', 
                       choices=['Regular', 'GradDotProd', 'InRunShapley'])
    
    # Architecture parameters
    parser.add_argument('--architecture', type=str, default='GPT2-Small',
                       choices=['GPT2-Small', 'GPT2-Medium', 'GPT2-Large'])
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size')
    parser.add_argument('--warmup_step', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    
    # Dataset parameters
    parser.add_argument('--train_set', type=str, default='pile')
    parser.add_argument('--val_set', type=str, default='pile')
    
    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--eval_iter', type=int, default=20)
    parser.add_argument('--eval_bs', type=int, default=16)
    
    # In-Run Shapley specific parameters
    parser.add_argument('--shapley_order', type=int, default=1, choices=[1, 2],
                       help='Order of Shapley approximation (1=first-order, 2=second-order)')
    parser.add_argument('--accumulate_shapley', action='store_true', default=True,
                       help='Accumulate Shapley values across iterations')
    parser.add_argument('--shapley_save_interval', type=int, default=10,
                       help='How often to save Shapley values')
    parser.add_argument('--dot_prod_save_interval', type=int, default=10,
                       help='How often to save gradient dot products')
    
    # Precision parameters
    parser.add_argument('--model_dtype', type=str, default='float32',
                       choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--train_dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'])
    
    # WandB logging
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='InRunShapley', 
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online',
                       choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_dir', type=str, default=None)
    parser.add_argument('--dynamic_val_batch', action='store_true',
                       help='Refresh validation batch every training step')
    parser.add_argument('--log_grad_norms', action='store_true',
                       help='Record per-sample training gradient norms')
    
    return parser.parse_args()


class InRunShapleyConfig:
    """Configuration class for In-Run Data Shapley."""
    
    def __init__(self, args):
        self.args = args
        
        # Model configuration
        self.architecture = args.architecture
        
        # Training hyperparameters
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.learning_rate = args.learning_rate
        self.min_lr = self.learning_rate * 0.1
        self.max_steps = args.max_steps
        self.seed = args.seed
        
        # Optimizer settings
        self.optimizer = args.optimizer
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        self.warmup_iters = args.warmup_step
        self.lr_decay_iters = 10000
        self.decay_lr = True
        
        # System settings
        self.device = 'cuda'
        self.compile = False
        self.backend = 'nccl'
        
        # Precision settings
        self.model_dtype = args.model_dtype
        self.train_dtype = args.train_dtype
        
        # Gradient accumulation
        self.full_batch_size = args.batch_size
        self.gradient_accumulation_steps = 1
        
        # Evaluation settings
        self.eval_iters = args.eval_iter
        self.eval_interval = args.eval_interval
        self.eval_bs = args.eval_bs
        
        # In-Run Shapley specific settings
        self.method = args.method
        self.shapley_order = args.shapley_order
        self.accumulate_shapley = args.accumulate_shapley
        self.shapley_save_interval = args.shapley_save_interval
        self.dot_prod_save_interval = args.dot_prod_save_interval
        
        # WandB settings
        self.use_wandb = args.wandb
        self.wandb_project = args.wandb_project
        if args.wandb_run_name:
            self.wandb_run_name = args.wandb_run_name
        else:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = (
                f"InRunShapley_{args.architecture}_order{args.shapley_order}_"
                f"bs{args.batch_size}_lr{args.learning_rate}_{current_time}"
            )
        self.wandb_mode = args.wandb_mode
        self.dynamic_val_batch = args.dynamic_val_batch
        self.log_grad_norms = args.log_grad_norms
        
        # Result directory setup
        self.result_folder = os.path.join(RESULTS_DIR, self.wandb_run_name)
        self.setup_result_directories()
        self.wandb_dir = args.wandb_dir or self.result_dir
    
    def setup_result_directories(self):
        """Create result directories."""
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder, exist_ok=True)
            print(f"Results folder '{self.result_folder}' was created.")
        
        self.result_dir = build_result_dir(self.result_folder, self.method, self.args)
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir, exist_ok=True)
            print(f"Results directory '{self.result_dir}' was created.")
    
    def get_result_file_path(self):
        """Get the result file path for storing training statistics."""
        return os.path.join(self.result_dir + '_results.json')
