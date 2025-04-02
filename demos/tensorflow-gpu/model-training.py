import tensorflow as tf
import numpy as np
import time
import os
import psutil
import platform
from datetime import datetime

# ====== Environment Information Function ======
def print_system_info():
    """Print detailed system and TensorFlow information for debugging"""
    print("\n" + "="*50)
    print("SYSTEM AND TENSORFLOW INFORMATION")
    print("="*50)
    
    # System information
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"CPU: {platform.processor()}")
    print(f"CPU cores: {psutil.cpu_count(logical=False)} (Physical), {psutil.cpu_count(logical=True)} (Logical)")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    # TensorFlow information
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    # Check if TensorFlow is built with CUDA
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Number of GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            
        # Get more GPU details if possible
        try:
            for i, gpu in enumerate(gpus):
                details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU {i} details: {details}")
        except:
            print("Could not get detailed GPU information")
    else:
        print("No GPUs available")

# ====== GPU Configuration ======
def configure_gpu(memory_limit=None, xla=True):
    """Configure GPU settings for optimal performance"""
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU available, will only use CPU")
        return False
    
    print(f"Found {len(gpus)} GPU(s)")
    
    try:
        for gpu in gpus:
            # Enable memory growth to avoid allocating all memory at once
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Optionally limit GPU memory
        if memory_limit is not None:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
            )
            print(f"Limited GPU memory to {memory_limit} MB")
        
        # Enable mixed precision (uses float16 where possible)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled (float16/float32)")
        
        # Enable XLA compilation for potentially faster performance
        if xla:
            tf.config.optimizer.set_jit(True)
            print("XLA JIT compilation enabled")
        
        return True
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        return False

# ====== Dataset Creation ======
def create_dataset(size="small"):
    """Create a synthetic dataset for benchmarking"""
    # Define dataset sizes
    if size == "tiny":
        num_samples = 1000
        num_features = 10
    elif size == "small":
        num_samples = 5000
        num_features = 20
    elif size == "medium":
        num_samples = 20000
        num_features = 50
    elif size == "large":
        num_samples = 50000
        num_features = 100
    else:  # large
        num_samples = 50000
        num_features = 100
    
    num_classes = 5
    
    print(f"Creating {size} dataset: {num_samples} samples, {num_features} features")
    
    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    
    # Split into train and test
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_one_hot[:train_size], y_one_hot[train_size:]
    
    return (X_train, y_train), (X_test, y_test), num_features, num_classes

# ====== Model Creation Functions ======
def create_dense_model(input_shape, num_classes, size="small"):
    """Create a simple dense neural network of configurable size"""
    if size == "tiny":
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif size == "small":
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif size == "medium":
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:  # large
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cnn_model(input_shape, num_classes, size="small"):
    """Create a 1D CNN model which often performs better on GPUs"""
    # Create input layer and reshape for CNN
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    x = tf.keras.layers.Reshape((input_shape, 1))(inputs)
    
    # CNN architecture based on size
    if size == "tiny":
        x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
    elif size == "small":
        x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
    elif size == "medium":
        x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
    else:  # large
        x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ====== Training and Benchmarking ======
def train_and_benchmark(device, model_type="dense", dataset_size="small", model_size="small", 
                        batch_size=64, epochs=3, warmup=True):
    """Train model on specified device with detailed benchmarking"""
    print("\n" + "="*50)
    print(f"BENCHMARK: {device}, {model_type.upper()} model, {dataset_size} dataset, {model_size} model size")
    print("="*50)
    
    # Set device context
    if device == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf_device = "/CPU:0"
    else:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        tf_device = "/GPU:0"
    
    # Create dataset
    (X_train, y_train), (X_test, y_test), num_features, num_classes = create_dataset(dataset_size)
    
    # Use the specified device
    with tf.device(tf_device):
        # Create model based on type and size
        if model_type == "dense":
            model = create_dense_model(num_features, num_classes, model_size)
        else:  # cnn
            model = create_cnn_model(num_features, num_classes, model_size)
        
        model.summary()
        
        # Perform warm-up run to compile kernels (if requested)
        if warmup and device == "GPU":
            print("\nPerforming GPU warm-up...")
            # Use a small subset of data for warm-up
            subset_size = min(100, len(X_train))
            model.fit(
                X_train[:subset_size], y_train[:subset_size],
                epochs=1, batch_size=32, verbose=0
            )
            print("Warm-up complete")
        
        # Record detailed timing for each epoch
        epoch_times = []
        training_start = time.time()
        
        # Custom callback to measure epoch times
        class TimingCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
            
            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                epoch_times.append(epoch_time)
                print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds")
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[TimingCallback()]
        )
        
        # Calculate total training time
        total_time = time.time() - training_start
        
        # Evaluate the model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Return comprehensive results
        results = {
            "device": device,
            "model_type": model_type,
            "dataset_size": dataset_size,
            "model_size": model_size,
            "batch_size": batch_size,
            "total_time": total_time,
            "epoch_times": epoch_times,
            "accuracy": test_acc,
            "loss": test_loss,
            "dataset_shape": X_train.shape,
            "parameters": model.count_params()
        }
        
        # Print summary
        print("\nBenchmark Results:")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.2f} seconds")
        print(f"Fastest epoch: {min(epoch_times):.2f} seconds")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Model parameters: {model.count_params():,}")
        
        return results

# ====== Main Benchmark Function ======
def run_comprehensive_benchmarks():
    """Run a series of benchmarks to identify optimal configuration"""
    print_system_info()
    configure_gpu()
    
    results = []
    
    # First try the tiniest model on both CPU and GPU to verify basic functionality
    cpu_tiny = train_and_benchmark("CPU", "dense", "tiny", "tiny", batch_size=32)
    results.append(cpu_tiny)
    
    # Check if we have GPU
    if tf.config.list_physical_devices('GPU'):
        gpu_tiny = train_and_benchmark("GPU", "dense", "tiny", "tiny", batch_size=32)
        results.append(gpu_tiny)
        
        # Calculate speedup
        cputime = cpu_tiny["total_time"]
        gputime = gpu_tiny["total_time"]
        print(f"Tiny CPU time: {cputime}")
        print(f"Tiny GPU time: {gputime}")
        speedup = cpu_tiny["total_time"] / gpu_tiny["total_time"]
        print(f"\nTiny model GPU vs CPU speedup: {speedup:.2f}x")
        
        # If GPU is working reasonably well, continue with larger models
        if speedup >= 0.5:  # If GPU is at least half as fast as CPU
            # Try CNN model which often works better on GPUs
            # cpu_cnn = train_and_benchmark("CPU", "cnn", "small", "small", batch_size=64)
            # results.append(cpu_cnn)
            
            # gpu_cnn = train_and_benchmark("GPU", "cnn", "small", "small", batch_size=64)
            # results.append(gpu_cnn)

            # cputime = cpu_cnn["total_time"]
            # gputime = gpu_cnn["total_time"]
            # print(f"Small CNN model CPU time: {cputime}")
            # print(f"Small CNN model GPU time: {gputime}")
            
            # cnn_speedup = cpu_cnn["total_time"] / gpu_cnn["total_time"]
            # print(f"\nSmall CNN model GPU vs CPU speedup: {cnn_speedup:.2f}x")
            
            # Try larger batch sizes which typically benefit GPUs
            # for batch_size in [128, 256, 512]:
            for batch_size in [512]:
                cpu_cnn = train_and_benchmark("CPU", "cnn", "large", "large", batch_size=batch_size)
                results.append(cpu_cnn)
                
                gpu_batch = train_and_benchmark("GPU", "cnn", "large", "large", 
                                               batch_size=batch_size)
                results.append(gpu_batch)

                cputime = cpu_cnn["total_time"]
                gputime = gpu_batch["total_time"]
                print(f"Batch size {batch_size} CPU time: {cputime}")
                print(f"Batch size {batch_size} GPU time: {gputime}")
            
                batch_speedup = cpu_cnn["total_time"] / gpu_batch["total_time"]
                print(f"\nBatch size {batch_size} GPU vs CPU speedup: {batch_speedup:.2f}x")
                
                # If we found a good configuration, try a medium model
                # if batch_speedup >= 1.0:
                #     gpu_medium = train_and_benchmark("GPU", "cnn", "medium", "medium", 
                #                                    batch_size=batch_size)
                #     results.append(gpu_medium)
                    # break
    else:
        print("No GPU available, skipping GPU benchmarks")
    
    # Print summary of all benchmarks
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    for i, r in enumerate(results):
        print(f"{i+1}. {r['device']} {r['model_type']} model ({r['model_size']} size, {r['dataset_size']} dataset)")
        print(f"   Batch size: {r['batch_size']}, Total time: {r['total_time']:.2f}s, Accuracy: {r['accuracy']:.4f}")
    
    return results

# For use in a notebook
if __name__ == "__main__" or '__file__' not in globals():
    try:
        # Try to import psutil for system information
        import psutil
    except ImportError:
        print("psutil not installed. Install with: pip install psutil")
        # Create a dummy psutil.cpu_count function
        class DummyPsutil:
            @staticmethod
            def cpu_count(logical=True):
                return "Unknown"
            
            class virtual_memory:
                total = "Unknown"
        
        psutil = DummyPsutil()
    
    # Run all benchmarks
    all_results = run_comprehensive_benchmarks()