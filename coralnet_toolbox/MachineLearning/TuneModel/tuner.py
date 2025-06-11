# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Module provides functionalities for hyperparameter tuning of the Ultralytics YOLO models for object detection, instance
segmentation, image classification, pose estimation, and multi-object tracking.

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters
that yield the best model performance. This is particularly crucial in deep learning models like YOLO,
where small changes in hyperparameters can lead to significant differences in model accuracy and efficiency.

Examples:
    Tune hyperparameters for YOLO11n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolo11n.pt")
    >>> model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
"""

import gc
import time
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from scipy.ndimage import gaussian_filter1d

import torch
from ultralytics import YOLO

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, YAML, callbacks, colorstr, remove_colorstr
from ultralytics.utils.plotting import plt_color_scatter


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Default hyperparameter search space containing bounds and scaling factors for mutation
DEFAULT_SPACE = {
    # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
    "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
    "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
    "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
    "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
    "box": (1.0, 20.0),  # box loss gain
    "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
    "dfl": (0.4, 6.0),  # dfl loss gain
    "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
    "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
    "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
    "degrees": (0.0, 180),  # image rotation (+/- deg)
    "translate": (0.0, 1.0),  # image translation (+/- fraction)
    "scale": (0.5, 2.0),  # image scale (+/- gain)
    "shear": (-180, 180),  # image shear (+/- deg)
    "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
    "flipud": (0.0, 1.0),  # image flip up-down (probability)
    "fliplr": (0.0, 1.0),  # image flip left-right (probability)
    "bgr": (0.0, 1.0),  # image channel bgr (probability)
    "mosaic": (0.0, 1.0),  # image mosaic (probability)
    "mixup": (0.0, 1.0),  # image mixup (probability)
    "cutmix": (0.0, 1.0),  # image cutmix (probability)
    "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
    "erasing": (0.0, 0.9),  # erasing probability
    "dropout": (0.0, 0.9),  # dropout probability
    "multi_scale": (0.0, 1.0),  # multi-scale training (probability)
}


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Tuner:
    """
    A class for hyperparameter tuning of YOLO models with advanced mutation strategies.

    The class evolves YOLO model hyperparameters over a given number of iterations by mutating them according to the
    search space and retraining the model to evaluate their performance.

    Attributes:
        space (Dict[str, tuple]): Hyperparameter search space containing bounds and scaling factors for mutation.
        tune_dir (Path): Directory where evolution logs and results will be saved.
        tune_csv (Path): Path to the CSV file where evolution logs are saved.
        args (dict): Configuration arguments for the tuning process.
        callbacks (list): Callback functions to be executed during tuning.
        prefix (str): Prefix string for logging messages.
        mutation_method (str): The mutation strategy to use.
        generation (int): Current generation counter for adaptive methods.
        population_history (List): History of successful hyperparameter sets.

    Methods:
        _mutate: Mutate hyperparameters based on bounds and scaling factors.
        __call__: Execute the hyperparameter evolution across multiple iterations.

    Examples:
        Tune hyperparameters for YOLO11n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> model.tune(
        ...     data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False,
        ...     mutation_method="adaptive"
        ... )

        Tune with custom search space and mutation method.
        >>> model.tune(space={key1: val1, key2: val2}, mutation_method="cauchy")  # custom search space dictionary
    """

    def __init__(self, mutation="gaussian", args=DEFAULT_CFG, _callbacks: Optional[List] = None):
        """
        Initialize the Tuner with configurations.

        Args:
            args (dict): Configuration for hyperparameter evolution.
            _callbacks (List, optional): Callback functions to be executed during tuning.
        """
        self.space = args.pop("space", None) or DEFAULT_SPACE  # key: (min, max)
        self.mutation_method = mutation
        
        # Load and merge configuration arguments
        self.args = get_cfg(overrides=args)
        # Set exist_ok to True if resuming, to allow overwriting the tune_dir
        self.args.exist_ok = self.args.resume  # resume w/ same tune_dir
        # Determine the directory to save tuning results
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
        # Reset name, exist_ok, and resume to avoid affecting subsequent training runs
        self.args.name, self.args.exist_ok, self.args.resume = (None, False, False)  # reset to not affect training
        # Path to the CSV file where tuning results will be logged
        self.tune_csv = self.tune_dir / "tune_results.csv"
        # Set up callback functions for the tuning process
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        # Prefix string for logging messages
        self.prefix = colorstr("Tuner: ")
        
        # Initialize mutation-specific attributes
        self.generation = 0
        self.population_history = []
        self.max_generations = 100  # Will be updated in __call__
        
        # Add integration callbacks for the tuner
        callbacks.add_integration_callbacks(self)
        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}Using mutation method: {self.mutation_method}\n"
            f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )

    def _mutate(
        self, parent: str = "single", n: int = 5, mutation: float = 0.8, sigma: float = 0.2
    ) -> Dict[str, float]:
        """
        Mutate hyperparameters using the specified mutation method.

        Args:
            parent (str): Parent selection method: 'single' or 'weighted'.
            n (int): Number of parents to consider.
            mutation (float): Probability of a parameter mutation in any given iteration.
            sigma (float): Standard deviation for Gaussian random number generator.

        Returns:
            (Dict[str, float]): A dictionary containing mutated hyperparameters.
        """
        # Get base hyperparameters
        base_hyp = self._get_base_hyperparameters(parent, n, mutation, sigma)
        
        # Apply mutation method
        if self.mutation_method == "gaussian":
            return self._mutate_gaussian(base_hyp, parent, n, mutation, sigma)
        elif self.mutation_method == "adaptive":
            return self._mutate_adaptive_gaussian(base_hyp, mutation)
        elif self.mutation_method == "cauchy":
            return self._mutate_cauchy(base_hyp, mutation)
        elif self.mutation_method == "polynomial":
            return self._mutate_polynomial(base_hyp, mutation)
        elif self.mutation_method == "levy":
            return self._mutate_levy_flight(base_hyp, mutation)
        elif self.mutation_method == "differential":
            return self._mutate_differential(base_hyp, mutation)
        elif self.mutation_method == "parameter_specific":
            return self._mutate_parameter_specific(base_hyp, mutation)
        elif self.mutation_method == "multi_scale":
            return self._mutate_multi_scale(base_hyp, mutation)
        elif self.mutation_method == "simulated_annealing":
            return self._mutate_simulated_annealing(base_hyp, mutation)
        else:
            LOGGER.warning(f"{self.prefix}Unknown mutation method '{self.mutation_method}', using gaussian")
            return self._mutate_gaussian(base_hyp, parent, n, mutation, sigma)

    def _get_base_hyperparameters(self, parent: str, n: int, mutation: float, sigma: float) -> Dict[str, float]:
        """Get base hyperparameters from existing results or initialize new ones."""
        if self.tune_csv.exists():
            return self._get_from_existing_results(parent, n, mutation, sigma)
        else:
            return self._initialize_continuous_params()

    def _get_from_existing_results(self, parent: str, n: int, mutation: float, sigma: float) -> Dict[str, float]:
        """Get base hyperparameters from existing CSV results."""
        # Load and select parent(s)
        x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
        fitness = x[:, 0]  # first column
        n = min(n, len(x))  # number of previous results to consider
        x = x[np.argsort(-fitness)][:n]  # top n mutations
        w = x[:, 0] - x[:, 0].min() + 1e-6  # weights (sum > 0)
        
        # Update population history for differential evolution
        if len(self.population_history) < 20:  # Keep last 20 for memory efficiency
            param_dict = {k: float(x[0, i + 1]) for i, k in enumerate(self.space.keys())}
            self.population_history.append(param_dict)
        
        # Select parent based on method
        if parent == "single" or len(x) == 1:
            x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
        elif parent == "weighted":
            x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

        # Return base hyperparameters
        return {k: float(x[i + 1]) for i, k in enumerate(self.space.keys())}

    def _initialize_continuous_params(self) -> Dict[str, float]:
        """Initialize continuous parameters from args for first iteration."""
        hyp = {k: getattr(self.args, k) for k in self.space.keys()}
        return self._constrain_continuous_params(hyp)

    def _constrain_continuous_params(self, hyp: Dict[str, float]) -> Dict[str, float]:
        """Apply bounds constraints to continuous parameters."""
        for k, v in self.space.items():
            hyp[k] = max(hyp[k], v[0])  # lower limit
            hyp[k] = min(hyp[k], v[1])  # upper limit
            hyp[k] = round(hyp[k], 5)   # significant digits
        return hyp

    # Original Gaussian mutation method
    def _mutate_gaussian(self, base_hyp: Dict, parent: str, n: int, mutation: float, sigma: float) -> Dict[str, float]:
        """Original Gaussian mutation method."""
        if not self.tune_csv.exists():
            return base_hyp
        
        # Initialize random generator
        r = np.random
        r.seed(int(time.time()))
        
        # Load existing results for mutation
        x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
        fitness = x[:, 0]
        n = min(n, len(x))
        x = x[np.argsort(-fitness)][:n]
        w = x[:, 0] - x[:, 0].min() + 1e-6
        
        if parent == "single" or len(x) == 1:
            x = x[random.choices(range(n), weights=w)[0]]
        elif parent == "weighted":
            x = (x * w.reshape(n, 1)).sum(0) / w.sum()

        # Apply mutations
        g = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])
        ng = len(self.space)
        v = np.ones(ng)
        
        while all(v == 1):
            v = (g * (r.random(ng) < mutation) * r.randn(ng) * r.random() * sigma + 1).clip(0.3, 3.0)
        
        hyp = {k: float(x[i + 1] * v[i]) for i, k in enumerate(self.space.keys())}
        return self._constrain_continuous_params(hyp)

    def _mutate_adaptive_gaussian(self, base_hyp: Dict, mutation: float) -> Dict[str, float]:
        """Adaptive Gaussian mutation with decreasing variance over generations."""
        initial_sigma = 0.3
        final_sigma = 0.05
        progress = self.generation / self.max_generations
        sigma = initial_sigma * (1 - progress) + final_sigma * progress
        
        mutated = base_hyp.copy()
        for k, v in self.space.items():
            if random.random() < mutation:
                current_val = base_hyp[k]
                param_range = v[1] - v[0]
                noise = np.random.normal(0, sigma * param_range)
                mutated[k] = current_val + noise
        
        return self._constrain_continuous_params(mutated)

    def _mutate_cauchy(self, base_hyp: Dict, mutation: float, scale: float = 0.1) -> Dict[str, float]:
        """Cauchy mutation with heavy tails for escaping local optima."""
        mutated = base_hyp.copy()
        for k, v in self.space.items():
            if random.random() < mutation:
                current_val = base_hyp[k]
                param_range = v[1] - v[0]
                noise = np.random.standard_cauchy() * scale * param_range
                mutated[k] = current_val + noise
        
        return self._constrain_continuous_params(mutated)

    def _mutate_polynomial(self, base_hyp: Dict, mutation: float, eta: float = 20.0) -> Dict[str, float]:
        """Polynomial mutation from genetic algorithms."""
        mutated = base_hyp.copy()
        for k, v in self.space.items():
            if random.random() < mutation:
                current_val = base_hyp[k]
                min_val, max_val = v[0], v[1]
                
                # Normalize to [0, 1]
                normalized = (current_val - min_val) / (max_val - min_val)
                
                u = random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                # Apply mutation and denormalize
                new_normalized = normalized + delta
                mutated[k] = min_val + new_normalized * (max_val - min_val)
        
        return self._constrain_continuous_params(mutated)

    def _mutate_levy_flight(self, base_hyp: Dict, mutation: float, alpha: float = 1.5, 
                            scale: float = 0.1) -> Dict[str, float]:
        """Lévy flight mutation for combined local and global search."""
        mutated = base_hyp.copy()
        for k, v in self.space.items():
            if random.random() < mutation:
                current_val = base_hyp[k]
                param_range = v[1] - v[0]
                
                # Generate Lévy flight step
                levy_step = self._levy_flight(alpha) * scale * param_range
                mutated[k] = current_val + levy_step
        
        return self._constrain_continuous_params(mutated)

    def _mutate_differential(self, base_hyp: Dict, mutation: float, F: float = 0.5) -> Dict[str, float]:
        """Differential Evolution mutation using population history."""
        if len(self.population_history) < 3:
            return self._mutate_adaptive_gaussian(base_hyp, mutation)  # fallback
        
        # Select three random individuals
        candidates = [p for p in self.population_history if p != base_hyp]
        if len(candidates) < 3:
            return base_hyp
        
        r1, r2, r3 = random.sample(candidates, 3)
        
        mutated = base_hyp.copy()
        for k, v in self.space.items():
            if random.random() < mutation:
                # DE mutation formula
                new_val = r1[k] + F * (r2[k] - r3[k])
                mutated[k] = new_val
        
        return self._constrain_continuous_params(mutated)

    def _mutate_parameter_specific(self, base_hyp: Dict, mutation: float) -> Dict[str, float]:
        """Parameter-specific mutation strategies."""
        mutated = base_hyp.copy()
        
        for k, v in self.space.items():
            if random.random() < mutation:
                current_val = base_hyp[k]
                
                # Parameter-specific mutation strategies
                if 'lr' in k.lower() or 'learning' in k.lower():
                    # Learning rates: log-normal mutation
                    log_val = np.log(max(current_val, 1e-8))
                    log_val += np.random.normal(0, 0.1)
                    new_val = np.exp(log_val)
                
                elif 'weight_decay' in k.lower() or 'decay' in k.lower():
                    # Weight decay: log-uniform mutation
                    log_min, log_max = np.log(max(v[0], 1e-8)), np.log(v[1])
                    new_val = np.exp(np.random.uniform(log_min, log_max))
                
                elif 'dropout' in k.lower() or 'prob' in k.lower():
                    # Probabilities: beta distribution mutation
                    alpha, beta = 2, 2  # favor middle values
                    new_val = np.random.beta(alpha, beta) * (v[1] - v[0]) + v[0]
                
                elif 'batch_size' in k.lower() or 'size' in k.lower():
                    # Sizes: geometric progression
                    sizes = [2**i for i in range(int(np.log2(max(v[0], 1))), int(np.log2(v[1])) + 1)]
                    new_val = random.choice(sizes) if sizes else current_val
                
                else:
                    # Default: Gaussian mutation
                    param_range = v[1] - v[0]
                    noise = np.random.normal(0, 0.1 * param_range)
                    new_val = current_val + noise
                
                mutated[k] = new_val
        
        return self._constrain_continuous_params(mutated)

    def _mutate_multi_scale(self, base_hyp: Dict, mutation: float, 
                            scales: List[float] = [0.01, 0.1, 0.3]) -> Dict[str, float]:
        """Multi-scale mutation with different step sizes."""
        mutated = base_hyp.copy()
        
        for k, v in self.space.items():
            if random.random() < mutation:
                current_val = base_hyp[k]
                param_range = v[1] - v[0]
                
                # Randomly choose a scale
                scale = random.choice(scales)
                noise = np.random.normal(0, scale * param_range)
                mutated[k] = current_val + noise
        
        return self._constrain_continuous_params(mutated)

    def _mutate_simulated_annealing(self, base_hyp: Dict, mutation: float) -> Dict[str, float]:
        """Temperature-based mutation with cooling schedule."""
        # Calculate temperature (starts high, decreases over generations)
        initial_temp = 1.0
        final_temp = 0.01
        progress = self.generation / self.max_generations
        temperature = initial_temp * (1 - progress) + final_temp * progress
        
        mutated = base_hyp.copy()
        for k, v in self.space.items():
            if random.random() < mutation:
                current_val = base_hyp[k]
                param_range = v[1] - v[0]
                
                # Step size proportional to temperature
                step_size = temperature * param_range * 0.1
                noise = np.random.normal(0, step_size)
                mutated[k] = current_val + noise
        
        return self._constrain_continuous_params(mutated)

    def _levy_flight(self, alpha: float) -> float:
        """Generate a Lévy flight step."""
        sigma_u = (np.math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / 
                   (np.math.gamma((1 + alpha) / 2) * alpha * (2 ** ((alpha - 1) / 2)))) ** (1 / alpha)
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, 1)
        return u / (abs(v) ** (1 / alpha))
    
    def prepare_base_model(self, base_model_save_path: str = "base_trained_model.pt") -> str:
        """
        Prepare a base trained model that will be used as the starting point for all hyperparameter iterations.
        This ensures fair comparison by having all iterations start from the same model state.
        
        Args:
            base_model_save_path: Path where the base trained model will be saved
            
        Returns:
            str: Path to the saved base model
        """        
        # Load the original model
        original_model_path = self.args.model
        original_model = YOLO(original_model_path)
        
        LOGGER.info(f"{self.prefix}Preparing base model from {original_model_path}")
        
        # Train the model with base settings to create a starting point
        base_train_args = {**vars(self.args)}
        # Remove model from train args to avoid conflicts
        base_train_args = {k: v for k, v in base_train_args.items() if k != 'model'}
        # Use minimal training for the base model (can be customized)
        base_train_args['epochs'] = 1
        base_train_args['resume'] = False
        
        LOGGER.info(f"{self.prefix}Training base model with settings: {base_train_args}")
        original_model.train(**base_train_args)
        
        # Save the trained base model
        base_model_path = Path(base_model_save_path)
        original_model.save(base_model_path)
        
        LOGGER.info(f"{self.prefix}Base model saved to {base_model_path}")
        return str(base_model_path)

    def __call__(self, base_model_path: str = None, iterations: int = 10, cleanup: bool = True, 
                 auto_prepare_base: bool = True):
        """
        Execute the hyperparameter evolution process when the Tuner instance is called.

        Args:
            base_model_path (str): Path to a base trained model that each iteration should start from.
                                 If None and auto_prepare_base is True, will create one automatically.
            iterations (int): The number of generations to run the evolution for.
            cleanup (bool): Whether to delete iteration weights to reduce storage space used during tuning.
            auto_prepare_base (bool): If True and base_model_path is None, automatically prepare a base model.
        """
        # Initialize tuning session
        t0 = time.time()
        best_save_dir, best_metrics = None, None
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)
        
        # Initialize timing tracking
        iteration_times = []
        
        # Set max generations for adaptive methods
        self.max_generations = iterations
        
        # Prepare base model if needed
        base_model_prep_start = time.time()
        if base_model_path is None and auto_prepare_base:
            base_model_path = self.prepare_base_model(self.tune_dir / "base_trained_model.pt")
            base_model_prep_time = time.time() - base_model_prep_start
            LOGGER.info(f"{self.prefix}Automatically prepared base model at {base_model_path} "
                        f"(took {base_model_prep_time:.2f}s)")
        elif base_model_path is not None:
            base_model_prep_time = 0
            LOGGER.info(f"{self.prefix}Using provided base model at {base_model_path}")
        else:
            base_model_prep_time = 0
            LOGGER.info(f"{self.prefix}No base model - using subprocess method for training")
        
        # Determine starting iteration (for resume functionality)
        start = self._get_starting_iteration()
        self.generation = start
        
        # Main tuning loop
        for i in range(start, iterations):
            iteration_start_time = time.time()
            self.generation = i
            LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} (method: {self.mutation_method})")
            
            # Step 1: Generate new hyperparameters
            mutated_hyp = self._mutate()
            LOGGER.info(f"{self.prefix}Hyperparameters: {mutated_hyp}")

            # Step 2: Train model with mutated hyperparameters
            training_start_time = time.time()
            if base_model_path is not None:
                # Use direct training with base model
                metrics, save_dir = self._train_with_hyperparameters_modified(mutated_hyp, i + 1, base_model_path)
            else:
                # Fallback to subprocess method if no base model provided
                metrics, save_dir = self._train_with_hyperparameters(mutated_hyp, i + 1)
            training_time = time.time() - training_start_time
            
            # Step 3: Log results to CSV
            self._log_results_to_csv(metrics, mutated_hyp)

            # Step 4: Track best results and cleanup
            best_save_dir, best_metrics = self._update_best_results(
                metrics, save_dir, best_save_dir, best_metrics, i, cleanup
            )

            # Calculate and log iteration timing
            iteration_end_time = time.time()
            iteration_total_time = iteration_end_time - iteration_start_time
            iteration_times.append(iteration_total_time)
            
            fitness = metrics.get("fitness", 0.0)
            LOGGER.info(f"{self.prefix}Iteration {i + 1} completed in {iteration_total_time:.2f}s "
                        f"(training: {training_time:.2f}s, fitness: {fitness:.4f})")

            # Step 5: Generate reports and plots
            self._generate_reports(i + 1, 
                                   iterations, 
                                   t0, 
                                   best_metrics, 
                                   best_save_dir, 
                                   iteration_times, 
                                   base_model_prep_time)
            
            # Clear cache to free memory
            gc.collect()
            torch.cuda.empty_cache()

    def _get_starting_iteration(self) -> int:
        """Get the starting iteration number for resume functionality."""
        if self.tune_csv.exists():
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            start = x.shape[0]
            LOGGER.info(f"{self.prefix}Resuming tuning run {self.tune_dir} from iteration {start + 1}...")
            return start
        return 0

    def _train_with_hyperparameters(self, mutated_hyp: Dict, iteration: int) -> tuple:
        """Train YOLO model with mutated hyperparameters."""
        metrics = {}
        
        # Convert boolean parameters from float to bool
        boolean_params = ['multi_scale']  # Add other boolean params here if needed
        for param in boolean_params:
            if param in mutated_hyp:
                mutated_hyp[param] = bool(round(mutated_hyp[param]))
        
        train_args = {**vars(self.args), **mutated_hyp}
        save_dir = get_save_dir(get_cfg(train_args))
        weights_dir = save_dir / "weights"
        
        try:
            # Train YOLO model with mutated hyperparameters (run in subprocess to avoid dataloader hang)
            launch = [__import__("sys").executable, "-m", "ultralytics.cfg.__init__"]  # workaround yolo not found
            cmd = [*launch, "train", *(f"{k}={v}" for k, v in train_args.items())]
            return_code = subprocess.run(cmd, check=True).returncode
            
            # Load training metrics
            ckpt_file = weights_dir / ("best.pt" if (weights_dir / "best.pt").exists() else "last.pt")
            metrics = torch.load(ckpt_file)["train_metrics"]
            assert return_code == 0, "training failed"
            
        except Exception as e:
            LOGGER.error(f"training failure for hyperparameter tuning iteration {iteration}\n{e}")
            
        return metrics, save_dir

    def _train_with_hyperparameters_modified(self, mutated_hyp: Dict, iteration: int, 
                                             base_model_path: str = None) -> tuple:
        """Train YOLO model with mutated hyperparameters using direct model training.
        
        Each iteration starts from the same base model state to ensure fair comparison of hyperparameters.
        
        Args:
            mutated_hyp: Dictionary of mutated hyperparameters
            iteration: Current iteration number
            base_model_path: Path to the base trained model that each iteration should start from
        """
        metrics = {}
        
        # Convert boolean parameters from float to bool
        boolean_params = ['multi_scale']  # Add other boolean params here if needed
        for param in boolean_params:
            if param in mutated_hyp:
                mutated_hyp[param] = bool(round(mutated_hyp[param]))
        
        # Prepare training arguments
        train_args = {**vars(self.args), **mutated_hyp}
        save_dir = get_save_dir(get_cfg(train_args))
        weights_dir = save_dir / "weights"
        
        try:
            # Always start from the same base model state for fair comparison
            if base_model_path and Path(base_model_path).exists():
                # Load from the base trained model (each iteration starts from same point)
                current_model = YOLO(base_model_path)
                LOGGER.info(f"Loading base model from {base_model_path} for iteration {iteration}")
            else:
                # Fallback: load from the original model path
                model_path = train_args.get('model', 'yolo11n.pt')
                current_model = YOLO(model_path)
                LOGGER.warning(f"Base model path not found, using {model_path} for iteration {iteration}")
            
            # Train the model directly with mutated hyperparameters
            # Remove 'model' from train_args to avoid conflicts
            train_kwargs = {k: v for k, v in train_args.items() if k != 'model'}
            # Ensure resume=False so we don't resume from previous training
            train_kwargs['resume'] = False
            
            results = current_model.train(**train_kwargs)
            
            # Extract metrics from training results
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            elif hasattr(results, 'metrics'):
                metrics = results.metrics
            else:
                # Fallback: try to load from saved checkpoint
                ckpt_file = weights_dir / ("best.pt" if (weights_dir / "best.pt").exists() else "last.pt")
                if ckpt_file.exists():
                    checkpoint = torch.load(ckpt_file, map_location='cpu')
                    metrics = checkpoint.get("train_metrics", {})
                else:
                    LOGGER.warning(f"Could not find checkpoint file for iteration {iteration}")
                    metrics = {"fitness": 0.0}
            
            # Ensure fitness is available
            if "fitness" not in metrics:
                # Calculate fitness from available metrics (this is model-dependent)
                # For classification: might use top1 accuracy
                # For detection: might use mAP
                if "metrics/accuracy_top1" in metrics:
                    metrics["fitness"] = metrics["metrics/accuracy_top1"]
                elif "metrics/mAP50-95" in metrics:
                    metrics["fitness"] = metrics["metrics/mAP50-95"]
                else:
                    # Fallback fitness calculation
                    metrics["fitness"] = 0.0
                    LOGGER.warning(f"Could not determine fitness for iteration {iteration}")
            
        except Exception as e:
            LOGGER.error(f"training failure for hyperparameter tuning iteration {iteration}\n{e}")
            metrics = {"fitness": 0.0}
            
        return metrics, save_dir

    def _log_results_to_csv(self, metrics: Dict, mutated_hyp: Dict):
        """Log fitness and hyperparameters to CSV file."""
        fitness = metrics.get("fitness", 0.0)
        
        # Prepare log row: fitness + continuous params only
        # Convert boolean values to numeric for CSV storage
        log_values = []
        for k in self.space.keys():
            value = mutated_hyp[k]
            # Convert boolean to numeric for CSV storage
            if isinstance(value, bool):
                log_values.append(1.0 if value else 0.0)
            else:
                log_values.append(value)
        
        log_row = [round(fitness, 5)] + log_values
        
        # Prepare headers (only for new CSV files)
        headers = ""
        if not self.tune_csv.exists():
            header_list = ["fitness"] + list(self.space.keys())
            headers = ",".join(header_list) + "\n"
        
        # Write to CSV
        with open(self.tune_csv, "a", encoding="utf-8") as f:
            f.write(headers + ",".join(map(str, log_row)) + "\n")

    def _update_best_results(self, metrics: Dict, save_dir, best_save_dir, best_metrics, 
                             current_idx: int, cleanup: bool) -> tuple:
        """Update best results and handle cleanup of iteration weights."""
        # Load current results to find best
        x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
        fitness = x[:, 0]  # first column
        best_idx = fitness.argmax()
        best_is_current = best_idx == current_idx
        
        if best_is_current:
            # Current iteration is the best so far
            best_save_dir = save_dir
            best_metrics = {k: round(v, 5) for k, v in metrics.items()}
            
            # Copy best weights to tune directory
            weights_dir = save_dir / "weights"
            for ckpt in weights_dir.glob("*.pt"):
                shutil.copy2(ckpt, self.tune_dir / "weights")
                
        elif cleanup:
            # Remove iteration weights to save storage space
            weights_dir = save_dir / "weights"
            shutil.rmtree(weights_dir, ignore_errors=True)
            
        return best_save_dir, best_metrics

    def _generate_reports(self, current_iter: int, total_iterations: int, start_time: float, 
                          best_metrics: Dict, best_save_dir, iteration_times: List[float] = None, 
                          base_model_prep_time: float = 0):
        """Generate plots and save best hyperparameters with timing information."""
        # Generate evolution plot
        plot_tune_results(self.tune_csv)
        
        # Load data for best results
        x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
        fitness = x[:, 0]
        best_idx = fitness.argmax()
        
        # Calculate timing statistics
        elapsed_time = time.time() - start_time
        timing_info = ""
        
        if iteration_times:
            avg_iteration_time = sum(iteration_times) / len(iteration_times)
            fastest_iteration = min(iteration_times)
            slowest_iteration = max(iteration_times)
            total_training_time = sum(iteration_times)
            overhead_time = elapsed_time - total_training_time - base_model_prep_time
            
            timing_info = (
                f"{self.prefix}Timing Summary:\n"
                f"{self.prefix}  Base model preparation: {base_model_prep_time:.2f}s\n"
                f"{self.prefix}  Total training time: {total_training_time:.2f}s\n"
                f"{self.prefix}  Average iteration time: {avg_iteration_time:.2f}s\n"
                f"{self.prefix}  Fastest iteration: {fastest_iteration:.2f}s\n"
                f"{self.prefix}  Slowest iteration: {slowest_iteration:.2f}s\n"
                f"{self.prefix}  Overhead time: {overhead_time:.2f}s\n"
            )
        
        # Create status header
        header = (
            f"{self.prefix}{current_iter}/{total_iterations} iterations complete ✅ ({elapsed_time:.2f}s)\n"
            f"{timing_info}"
            f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}\n"
            f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n"
            f"{self.prefix}Best fitness metrics are {best_metrics}\n"
            f"{self.prefix}Best fitness model is {best_save_dir}\n"
            f"{self.prefix}Best fitness hyperparameters are printed below.\n"
        )
        LOGGER.info("\n" + header)
        
        # Prepare best hyperparameters data
        data = {}
        
        # Add continuous parameters
        for i, k in enumerate(self.space.keys()):
            value = float(x[best_idx, i + 1])
            # Convert boolean parameters back to bool for YAML output
            if k in ['multi_scale']:  # Add other boolean params here if needed
                data[k] = bool(round(value))
            else:
                data[k] = value
        
        # Save and print best hyperparameters
        yaml_header = remove_colorstr(header.replace(self.prefix, "# ")) + "\n"
        YAML.save(self.tune_dir / "best_hyperparameters.yaml", data=data, header=yaml_header)
        YAML.print(self.tune_dir / "best_hyperparameters.yaml")
        
        
# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
        
        
def plot_tune_results(csv_file: str = "tune_results.csv"):
    """
    Plot the evolution results stored in a 'tune_results.csv' file. The function generates a scatter plot for each key
    in the CSV, color-coded based on fitness scores. The best-performing configurations are highlighted on the plots.

    Args:
        csv_file (str, optional): Path to the CSV file containing the tuning results.

    Examples:
        >>> plot_tune_results("path/to/tune_results.csv")
    """

    def _save_one_file(file):
        """Save one matplotlib plot to 'file'."""
        # Save the current plot to file with 200 dpi
        plt.savefig(file, dpi=200)
        # Close the current plot to free memory
        plt.close()
        # Log the save action
        LOGGER.info(f"Saved {file}")

    # Convert csv_file to a Path object
    csv_file = Path(csv_file)
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)
    # Number of metric columns (fitness is the first column)
    num_metrics_columns = 1
    # Extract hyperparameter keys (column names) excluding the fitness column
    keys = [x.strip() for x in data.columns][num_metrics_columns:]
    # Convert DataFrame to numpy array for easier indexing
    x = data.values
    # Extract fitness values (first column)
    fitness = x[:, 0]
    # Find the index of the best fitness value
    j = np.argmax(fitness)
    # Calculate the number of rows/columns for subplot grid
    n = math.ceil(len(keys) ** 0.5)
    
    # Create a new figure for scatter plots
    plt.figure(figsize=(10, 10), tight_layout=True)
    # Iterate over each hyperparameter key
    for i, k in enumerate(keys):
        # Extract values for this hyperparameter
        v = x[:, i + num_metrics_columns]
        # Value of this hyperparameter for the best fitness
        mu = v[j]
        # Create a subplot
        plt.subplot(n, n, i + 1)
        # Scatter plot colored by fitness
        plt_color_scatter(v, fitness, cmap="plasma", alpha=0.8, edgecolors="none")
        # Mark the best point with a plus
        plt.plot(mu, fitness.max(), "k+", markersize=15)
        # Set subplot title
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})
        # Set axis label size
        plt.tick_params(axis="both", labelsize=8)
        # Hide y-ticks for non-first column subplots
        if i % n != 0:
            plt.yticks([])  
            
    # Save the scatter plots figure
    _save_one_file(csv_file.with_name("tune_scatter_plots.png"))

    # Prepare x-axis values for fitness vs iteration plot
    x_iter = range(1, len(fitness) + 1)
    # Create a new figure for fitness vs iteration
    plt.figure(figsize=(10, 6), tight_layout=True)
    # Plot raw fitness values per iteration
    plt.plot(x_iter, fitness, marker="o", linestyle="none", label="fitness")
    # Plot smoothed fitness curve using Gaussian filter
    plt.plot(x_iter, gaussian_filter1d(fitness, sigma=3), ":", label="smoothed", linewidth=2)
    # Set plot title and axis labels
    plt.title("Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    # Show grid
    plt.grid(True)
    # Show legend
    plt.legend()
    
    # Save the fitness vs iteration figure
    _save_one_file(csv_file.with_name("tune_fitness.png"))
    
    # Create parallel coordinates plot for parameters with variation only
    # Use all hyperparameters for visualization
    param_data = x[:, num_metrics_columns:num_metrics_columns + len(keys)]
    
    # Filter parameters that have actual variation
    param_mins = np.min(param_data, axis=0)
    param_maxs = np.max(param_data, axis=0)
    
    # Only include parameters where max != min (i.e., there's variation)
    varied_indices = []
    selected_keys = []
    for i, (param_name, param_min, param_max) in enumerate(zip(keys, param_mins, param_maxs)):
        if param_max > param_min:  # Parameter has variation
            varied_indices.append(i)
            selected_keys.append(param_name)
    
    # If no parameters have variation, skip the parallel coordinates plot
    if not varied_indices:
        LOGGER.info("No parameter variations found - skipping parallel coordinates plot")
        return
    
    # Extract only the varied parameters
    selected_data = param_data[:, varied_indices]
    selected_mins = param_mins[varied_indices]
    selected_maxs = param_maxs[varied_indices]

    # Create figure for parallel coordinates plot with larger size
    fig, ax = plt.subplots(figsize=(max(18, 2 * len(selected_keys)), 10))

    # Normalize each parameter to [0, 1] for plotting
    normalized_data = np.zeros_like(selected_data)

    for i in range(len(selected_keys)):
        normalized_data[:, i] = (selected_data[:, i] - selected_mins[i]) / (selected_maxs[i] - selected_mins[i])

    # Create colormap based on fitness values
    if fitness.max() > fitness.min():
        fitness_normalized = (fitness - fitness.min()) / (fitness.max() - fitness.min())
    else:
        fitness_normalized = np.ones_like(fitness) * 0.5
    colors = plt.cm.plasma(fitness_normalized)

    # Plot each trial as a line connecting its normalized parameter values
    x_positions = np.arange(len(selected_keys))
    for i in range(len(normalized_data)):
        ax.plot(x_positions, normalized_data[i], color=colors[i], alpha=0.6, linewidth=0.8)

    # Highlight the best performing trial
    best_normalized = normalized_data[j]
    ax.plot(x_positions, best_normalized, color='orange', linewidth=3, alpha=0.9,
            label=f'Best (fitness={fitness[j]:.3f})')

    # Set up the plot
    ax.set_xlim(-0.1, len(selected_keys) - 0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([])  # Remove parameter names from x-axis tick labels
    ax.grid(True, alpha=0.3)

    # Move the title higher
    ax.set_title('Hyperparameter Tuning - Parallel Coordinates Plot', fontsize=14, pad=60)

    # Move the legend to the upper left corner inside the axes to reduce whitespace
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), borderaxespad=0.2)

    # Create custom y-axis labels for each parameter showing actual value ranges
    for i, (param_name, param_min, param_max) in enumerate(zip(selected_keys, selected_mins, selected_maxs)):
        # Add parameter name below the axis (lower than before)
        ax.text(i, -0.18, param_name, ha='center', va='top', fontweight='bold', fontsize=10)

        # Add min value at bottom
        ax.text(i, -0.08, f'{param_min:.2g}', ha='center', va='top', fontsize=9, color='gray')

        # Add max value at top
        ax.text(i, 1.02, f'{param_max:.2g}', ha='center', va='bottom', fontsize=9, color='gray')

        # Add some intermediate tick marks
        for tick_pos in [0.25, 0.5, 0.75]:
            tick_value = param_min + tick_pos * (param_max - param_min)
            ax.text(i,
                    tick_pos,
                    f'{tick_value:.2g}',
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='lightgray',
                    alpha=0.8)

            # Add small tick mark
            ax.plot([i - 0.02, i + 0.02], [tick_pos, tick_pos], color='lightgray', linewidth=0.5)

    # Remove default y-axis ticks and labels since we have custom ones
    ax.set_yticks([])
    ax.set_ylabel('')

    # Add vertical lines at each parameter position for better visual separation
    for i in x_positions:
        ax.axvline(x=i, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5)

    # Create colorbar to show fitness scale with tighter spacing
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=fitness.min(), vmax=fitness.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02, aspect=30)
    cbar.set_label('Fitness', rotation=270, labelpad=20)

    plt.tight_layout()

    # Save the parallel coordinates plot
    _save_one_file(csv_file.with_name("tune_parallel_coordinates.png"))
