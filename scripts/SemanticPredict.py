import torch
import numpy as np
import time
import argparse
import os
import sys

# Add the scripts directory to path to import from SemanticTrain
sys.path.append(os.path.dirname(__file__))

try:
    # Import the new all-in-one class
    from SemanticTrain import SemanticModel 
    import segmentation_models_pytorch as smp
except ImportError:
    print("Warning: Could not import SemanticModel from SemanticTrain.py. Profiling will not be available.")
    SemanticModel = None
    smp = None


def profile_model(model, image_sizes=None, batch_sizes=None, num_runs=10):
    """
    Profile the model's performance with different image sizes and batch sizes.

    Args:
        model (SemanticModel): The model instance to profile.
        image_sizes (list of tuples): List of (height, width) tuples for image sizes.
        batch_sizes (list of int): List of batch sizes to test.
        num_runs (int): Number of runs to average for each configuration.
    """
    if image_sizes is None:
        image_sizes = [(640, 640), (512, 512), (1024, 1024)]
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]

    print("🤖 Profiling Model.predict() Performance")
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
            # Create dummy images
            images = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(bs)]

            # Warm-up run (uses the `imgsz` parameter)
            print("   • Warming up...")
            if bs == 1:
                _ = model.predict(images[0], imgsz=h) # Pass imgsz
            else:
                _ = model.predict(images, imgsz=h) # Pass imgsz
            
            print("   • Running benchmark...")
            for run in range(num_runs):
                start_time = time.perf_counter()
                if bs == 1:
                    _ = model.predict(images[0], imgsz=h)
                else:
                    _ = model.predict(images, imgsz=h)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            print(f"   📊 Result: {avg_time:.4f}s avg ({num_runs} runs)")
            print()

    print("=" * 50)


def main():
    """Parse command line arguments and run model profiling."""
    print("\n" + "=" * 60)
    print("🚀 SEMANTIC SEGMENTATION PREDICTOR PROFILING")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Semantic Segmentation Predictor Profiling')

    # Model options
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pre-trained model weights (.pt file).')

    # Predictor options
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (model will be loaded to this device).')

    parser.add_argument('--use_fp16', action='store_true',
                        help='Use FP16 precision for faster inference')

    parser.add_argument('--compile_model', action='store_true', default=False,
                        help='Compile model for optimized inference (PyTorch 2.0+).')

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
    print(f"   • Model path: {args.model_path}")
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
        # Load model
        print("🔧 Loading model...")
        if SemanticModel is None:
            raise Exception("SemanticModel not available. Check SemanticTrain.py.")
            
        model = SemanticModel(model_path=args.model_path)
        model.device = args.device  # Override device if specified
        
        print("   ✅ Model loaded successfully")

        # Run profiling
        print("\n⏱️ Starting profiling...")
        
        print("   • Initializing predictor with specified optimizations...")
        # Manually call predict once to trigger optimization
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _ = model.predict(
            dummy_img,
            use_fp16=args.use_fp16,
            compile_model=args.compile_model,
            use_int8=args.use_int8,
            imgsz=640
        )
        print("   • Predictor initialized. Starting benchmark...")
        
        profile_model(
            model=model,
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