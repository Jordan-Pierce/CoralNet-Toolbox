import torch
import numpy as np
import time
import argparse
import os
import sys

# Add the scripts directory to path to import from SemanticTrain
sys.path.append(os.path.dirname(__file__))

try:
    from SemanticTrain import ModelBuilder
    import segmentation_models_pytorch as smp
except ImportError:
    print("Warning: Could not import from SemanticTrain. Model building will not be available.")
    ModelBuilder = None
    smp = None


class Predictor:
    """Handles inference predictions on RGB images with confidence thresholding."""

    def __init__(self, model, device='cuda', use_fp16=True, compile_model=True, use_int8=False):
        """Initialize Predictor with model and optimization settings."""
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.use_int8 = use_int8 and torch.cuda.is_available()

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Compile model for faster inference (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                print("🚀 Model compiled for optimized inference")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")

        # Use half precision if requested (not compatible with INT8)
        if self.use_fp16 and not self.use_int8:
            self.model = self.model.half()
            print("⚡ Using FP16 precision for faster inference")

        # Use INT8 quantization if requested (highest priority optimization)
        if self.use_int8:
            self._quantize_model()
            print("🔢 Using INT8 quantization for maximum inference speed")
        elif self.use_fp16:
            print("⚡ Using FP16 precision for faster inference")

        # Pre-allocate reusable tensors for common image sizes
        self._preallocate_tensors()

    def _quantize_model(self):
        """Apply INT8 quantization to the model for faster inference."""
        try:
            # Set model to eval mode for quantization
            self.model.eval()

            # Configure quantization
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

            # Prepare model for quantization
            torch.quantization.prepare(self.model, inplace=True)

            # Calibrate with dummy data (required for static quantization)
            self._calibrate_quantization()

            # Convert to quantized model
            torch.quantization.convert(self.model, inplace=True)

            print("✅ Model successfully quantized to INT8")

        except Exception as e:
            print(f"⚠️ INT8 quantization failed, falling back to FP32: {e}")
            self.use_int8 = False
            # Reset model to CPU for fallback
            self.model.to('cpu')

    def _calibrate_quantization(self):
        """Calibrate quantization with representative data."""
        # Create dummy calibration data
        dummy_input = torch.randn(1, 3, 640, 640)

        # Run a few forward passes for calibration
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

    def _preallocate_tensors(self):
        """Pre-allocate commonly used tensors to avoid repeated allocations."""
        # Common image sizes for pre-allocation
        common_sizes = [(640, 640), (512, 512), (1024, 1024)]

        self.preallocated_tensors = {}
        for h, w in common_sizes:
            # RGB input tensor
            input_tensor = torch.empty(1, 3, h, w, dtype=torch.float16 if self.use_fp16 else torch.float32)
            input_tensor = input_tensor.pin_memory() if torch.cuda.is_available() else input_tensor
            self.preallocated_tensors[(h, w)] = input_tensor

    def predict(self, image, confidence_threshold=0.5):
        """
        Make prediction on an RGB numpy array image with optimized performance.

        Args:
            image (np.ndarray): RGB image as numpy array with shape (H, W, 3)
            confidence_threshold (float): Confidence threshold for predictions.

        Returns:
            np.ndarray: Predicted mask as numpy array with shape (H, W)
        """
        h, w = image.shape[:2]

        # Try to use pre-allocated tensor for common sizes
        if (h, w) in self.preallocated_tensors:
            input_tensor = self.preallocated_tensors[(h, w)].to(self.device, non_blocking=True)
            # Copy image data to pre-allocated tensor
            np_tensor = torch.from_numpy(image).float()
            if self.use_fp16 and not self.use_int8:
                np_tensor = np_tensor.half()
            # For INT8, keep as float32 since quantized models handle conversion
            input_tensor.copy_(np_tensor.permute(2, 0, 1).unsqueeze(0), non_blocking=True)
        else:
            # Fallback: create tensor on-the-fly (slower)
            input_tensor = torch.from_numpy(image).float()
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
            if self.use_fp16 and not self.use_int8:
                input_tensor = input_tensor.half()
            # For INT8, keep as float32 since quantized models handle conversion
            input_tensor = input_tensor.to(self.device, non_blocking=True)

        with torch.no_grad():
            # Get model predictions (probabilities)
            pred_logits = self.model(input_tensor)

            # Convert to probabilities
            pred_probs = torch.softmax(pred_logits, dim=1)

            # Get predicted class and confidence
            pred_confidence, pred_class = torch.max(pred_probs, dim=1)

            # Apply confidence threshold - set low confidence predictions to background (0)
            pred_class = torch.where(pred_confidence >= confidence_threshold,
                                     pred_class,
                                     torch.zeros_like(pred_class))

        # Convert to numpy and remove batch dimension
        mask = pred_class.squeeze(0).cpu().numpy()

        return mask

    def predict_batch(self, images, confidence_threshold=0.5):
        """
        Make predictions on a batch of RGB images for maximum throughput.

        Args:
            images (list of np.ndarray): List of RGB images as numpy arrays
            confidence_threshold (float): Confidence threshold for predictions.

        Returns:
            list of np.ndarray: List of predicted masks
        """
        if not images:
            return []

        # Assume all images are the same size for batching
        h, w = images[0].shape[:2]

        # Create batch tensor
        batch_size = len(images)
        batch_tensor = torch.empty(batch_size, 3, h, w,
                                   dtype=torch.float32)  # Always use float32 for quantized models

        # Fill batch tensor
        for i, img in enumerate(images):
            img_tensor = torch.from_numpy(img).float()
            if self.use_fp16 and not self.use_int8:
                img_tensor = img_tensor.half()
            # For INT8, keep as float32 since quantized models handle conversion
            batch_tensor[i] = img_tensor.permute(2, 0, 1)

        batch_tensor = batch_tensor.to(self.device, non_blocking=True)

        with torch.no_grad():
            # Get model predictions (probabilities)
            pred_logits = self.model(batch_tensor)

            # Convert to probabilities
            pred_probs = torch.softmax(pred_logits, dim=1)

            # Get predicted class and confidence
            pred_confidence, pred_class = torch.max(pred_probs, dim=1)

            # Apply confidence threshold - set low confidence predictions to background (0)
            pred_class = torch.where(pred_confidence >= confidence_threshold,
                                     pred_class,
                                     torch.zeros_like(pred_class))

        # Convert to numpy and split into individual masks
        masks = []
        for i in range(batch_size):
            mask = pred_class[i].cpu().numpy()
            masks.append(mask)

        return masks


def profile_predictor(predictor, image_sizes=None, batch_sizes=None, num_runs=10):
    """
    Profile the predictor's performance with different image sizes and batch sizes.

    Args:
        predictor (Predictor): The predictor instance to profile.
        image_sizes (list of tuples): List of (height, width) tuples for image sizes.
        batch_sizes (list of int): List of batch sizes to test.
        num_runs (int): Number of runs to average for each configuration.
    """
    if image_sizes is None:
        image_sizes = [(640, 640), (512, 512), (1024, 1024)]
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]

    print("🤖 Profiling Predictor Performance")
    print("=" * 50)
    print(f"   • Testing {len(image_sizes)} image sizes: {[f'{h}x{w}' for h,w in image_sizes]}")
    print(f"   • Testing {len(batch_sizes)} batch sizes: {batch_sizes}")
    print(f"   • Runs per configuration: {num_runs}")
    print()

    total_configs = len(image_sizes) * len(batch_sizes)
    config_count = 0

    for h, w in image_sizes:
        for bs in batch_sizes:
            config_count += 1
            print(f"🔬 Config {config_count}/{total_configs}: Image {h}x{w}, Batch {bs}")

            times = []
            for run in range(num_runs):
                # Create dummy images
                images = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(bs)]

                start_time = time.perf_counter()

                if bs == 1:
                    _ = predictor.predict(images[0])
                else:
                    _ = predictor.predict_batch(images)

                end_time = time.perf_counter()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            print(f"   📊 Result: {avg_time:.4f}s avg ({num_runs} runs)")
            print()

    print("=" * 50)


def main():
    """Parse command line arguments and run predictor profiling."""
    print("\n" + "=" * 60)
    print("🚀 SEMANTIC SEGMENTATION PREDICTOR PROFILING")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Semantic Segmentation Predictor Profiling')

    # Model options
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model weights (.pt file). If not provided, will build a new model.')

    parser.add_argument('--encoder_name', type=str, default='resnet34',
                        help='Encoder to use when building a new model (ignored if --model_path is provided)')

    parser.add_argument('--decoder_name', type=str, default='Unet',
                        help='Decoder to use when building a new model (ignored if --model_path is provided)')

    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes for new model (ignored if --model_path is provided)')

    # Predictor options
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')

    parser.add_argument('--use_fp16', action='store_true',
                        help='Use FP16 precision for faster inference')

    parser.add_argument('--compile_model', action='store_true', default=False,
                        help='Compile model for optimized inference (PyTorch 2.0+). '
                             'Requires C++ compiler.')

    parser.add_argument('--use_int8', action='store_true',
                        help='Use INT8 quantization for maximum inference speed')

    # Profiling options
    parser.add_argument('--image_sizes', type=str, default='640x640,512x512,1024x1024',
                        help='Comma-separated list of image sizes to test (format: HxW,HxW)')

    parser.add_argument('--batch_sizes', type=str, default='1,4,8',
                        help='Comma-separated list of batch sizes to test')

    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs to average for each configuration')

    args = parser.parse_args()

    print("📋 Parsed command line arguments")
    print(f"   • Model path: {args.model_path or 'None (will build new model)'}")
    print(f"   • Encoder: {args.encoder_name}")
    print(f"   • Decoder: {args.decoder_name}")
    print(f"   • Device: {args.device}")
    print(f"   • FP16: {'Enabled' if args.use_fp16 else 'Disabled'}")
    print(f"   • INT8: {'Enabled' if args.use_int8 else 'Disabled'}")
    print(f"   • Compile: {'Enabled' if args.compile_model else 'Disabled'}")
    print(f"   • Image sizes: {args.image_sizes}")
    print(f"   • Batch sizes: {args.batch_sizes}")
    print(f"   • Runs per config: {args.num_runs}")
    print()

    # Parse image sizes
    print("🔍 Parsing image sizes...")
    image_sizes = []
    for size_str in args.image_sizes.split(','):
        h, w = size_str.split('x')
        image_sizes.append((int(h), int(w)))
    print(f"   • Image sizes: {image_sizes}")

    # Parse batch sizes
    print("🔍 Parsing batch sizes...")
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    print(f"   • Batch sizes: {batch_sizes}")
    print()

    try:
        # Load or build model
        print("🔧 Loading/building model...")
        if args.model_path:
            if not os.path.exists(args.model_path):
                raise Exception(f"Model path does not exist: {args.model_path}")
            print(f"   • Loading model from {args.model_path}")
            model = torch.load(args.model_path, weights_only=False)
            print("   ✅ Model loaded successfully")
        else:
            if ModelBuilder is None:
                raise Exception("ModelBuilder not available. Please provide --model_path "
                                "or ensure SemanticTrain.py is accessible.")
            print(f"   • Building new model: {args.encoder_name} -> {args.decoder_name}")
            model_builder = ModelBuilder(
                encoder_name=args.encoder_name,
                decoder_name=args.decoder_name,
                num_classes=args.num_classes
            )
            model = model_builder.model
            print("   ✅ Model built successfully")

        # Create predictor
        print("\n🤖 Creating predictor...")
        predictor = Predictor(
            model=model,
            device=args.device,
            use_fp16=args.use_fp16,
            compile_model=args.compile_model,
            use_int8=args.use_int8
        )
        print("   ✅ Predictor created successfully")

        # Run profiling
        print("\n⏱️ Starting profiling...")
        profile_predictor(
            predictor=predictor,
            image_sizes=image_sizes,
            batch_sizes=batch_sizes,
            num_runs=args.num_runs
        )

        print("\n✅ Profiling completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
