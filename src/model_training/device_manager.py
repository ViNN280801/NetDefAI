import os
import sys
import torch
import platform
import multiprocessing
from enum import Enum, auto
from logger import device_manager_logger as logger


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class ComputeDevice(Enum):
    """Enum for different compute devices available for training."""

    CUDA = auto()  # NVIDIA GPU
    MPS = auto()  # Apple Metal Performance Shaders (M1/M2 GPU)
    XPU = auto()  # Intel GPUs
    CPU_MULTI = auto()  # Multiprocessing CPU
    CPU_SERIAL = auto()  # Single core CPU


class DeviceManager:
    """Manages the detection and selection of compute devices for model training."""

    def __init__(self):
        self.device_type = None
        self.available_devices = []
        self.num_cores = multiprocessing.cpu_count()
        self._detect_devices()

    def _detect_devices(self):
        """Detect available compute devices on the system."""
        self.available_devices = []

        # Check for CUDA (NVIDIA GPUs)
        try:
            import torch

            if torch.cuda.is_available():
                self.available_devices.append(ComputeDevice.CUDA)
                logger.info(
                    f"Found CUDA device: {
                        torch.cuda.get_device_name(0)}"
                )
                for i in range(torch.cuda.device_count()):
                    logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        except (ImportError, AttributeError):
            logger.debug("PyTorch with CUDA support not available")

        # Check for Apple Metal (MPS)
        try:
            import torch

            if (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.available_devices.append(ComputeDevice.MPS)
                logger.info("Found Apple Metal GPU (MPS)")
        except (ImportError, AttributeError):
            logger.debug("PyTorch with MPS support not available")

        # Check for Intel XPU
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                self.available_devices.append(ComputeDevice.XPU)
                logger.info("Found Intel XPU device")
        except (ImportError, AttributeError):
            logger.debug("PyTorch with Intel XPU support not available")

        # Check for TensorFlow GPU support
        try:
            # Import TensorFlow conditionally to avoid errors when not available
            tf = None
            try:
                import tensorflow as tf  # type: ignore [python3.13.2 doesn't support tensorflow]
            except ImportError:
                logger.debug("TensorFlow not available")

            if tf is not None:
                gpus = tf.config.list_physical_devices("GPU")
                if gpus and ComputeDevice.CUDA not in self.available_devices:
                    self.available_devices.append(ComputeDevice.CUDA)
                    logger.info(f"Found TensorFlow-compatible GPU devices: {len(gpus)}")
                    for gpu in gpus:
                        logger.info(f"  {gpu.name}")
        except AttributeError:
            logger.debug("TensorFlow GPU support not available")

        # CPU is always available
        if self.num_cores > 1:
            self.available_devices.append(ComputeDevice.CPU_MULTI)
            logger.info(
                f"Multiprocessing available with {
                    self.num_cores} CPU cores"
            )

        # Serial processing is always available as fallback
        self.available_devices.append(ComputeDevice.CPU_SERIAL)
        logger.info(f"Serial processing available on {platform.processor()}")

    def get_device_info(self):
        """
        Get detailed information about the selected device.

        Returns:
            dict: Device information
        """
        info = {
            "device_type": self.device_type.name if self.device_type else "None",
            "num_cores": self.num_cores,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

        # Add GPU-specific information if available
        if self.device_type == ComputeDevice.CUDA:
            try:
                import torch

                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_name"] = torch.cuda.get_device_name(0)
                # Get CUDA version using the correct attribute or alternative method
                try:
                    info["cuda_version"] = torch.__version__
                except AttributeError:
                    info["cuda_version"] = str(torch.cuda.get_device_capability(0))
            except (ImportError, AttributeError):
                try:
                    # Import TensorFlow conditionally to avoid errors when not available
                    tf = None
                    try:
                        import tensorflow as tf  # type: ignore [python3.13.2 doesn't support tensorflow]
                    except ImportError:
                        pass

                    if tf is not None:
                        gpus = tf.config.list_physical_devices("GPU")
                        info["gpu_count"] = len(gpus)
                        info["gpu_names"] = [gpu.name for gpu in gpus]
                except (ImportError, AttributeError):
                    pass

        return info

    def print_device_info(self):
        """Print information about the available compute devices."""
        print("\n===== COMPUTE DEVICE INFORMATION =====")
        print(f"Device type: {self.active_device.name}")
        print(f"Platform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
        print(f"CPU cores: {self.num_cores}")

        # Add GPU information
        try:
            if torch.cuda.is_available():
                print("\nGPU Information:")
                print("CUDA available: Yes")
                print(f"CUDA version: {torch.version.cuda}")  # type: ignore[has-attribute]
                print(f"GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                    props = torch.cuda.get_device_properties(i)
                    print(f"      Total memory: {props.total_memory / 1024**3:.2f} GB")
                    print(f"      CUDA capability: {props.major}.{props.minor}")
                    print(f"      Multi-processor count: {props.multi_processor_count}")
            else:
                print("\nGPU Information: No CUDA-compatible GPU detected")
        except Exception as e:
            print(f"\nError getting GPU information: {e}")

        print("=======================================\n")

        # Log the same information
        logger.info(
            f"Device type: {self.active_device.name}"
            + f"Platform: {platform.platform()}"
            + f"Processor: {platform.processor()}"
            + f"CPU cores: {self.num_cores}"
        )

        try:
            if torch.cuda.is_available():
                logger.info("CUDA available: Yes")
                logger.info(f"CUDA version: {torch.version.cuda}")  # type: ignore[has-attribute]
                logger.info(f"GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
                    logger.info(f"CUDA capability: {props.major}.{props.minor}")
            else:
                logger.info("GPU Information: No CUDA-compatible GPU detected")
        except Exception as e:
            logger.warning(f"Error getting GPU information: {e}")

    def select_device(self, preferred_device=None):
        """
        Select the compute device based on the user's preference and availability.

        Args:
            preferred_device: Preferred compute device (from ComputeDevice enum)

        Returns:
            selected_device: Selected compute device (from ComputeDevice enum)
        """
        # Use the user's preference if specified
        if preferred_device is not None:
            if preferred_device == ComputeDevice.CUDA:
                # Check if CUDA is available
                if not torch.cuda.is_available():
                    logger.warning(
                        "CUDA requested but not available, falling back to CPU"
                    )
                    preferred_device = ComputeDevice.CPU_MULTI
                else:
                    logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")

            # Use the preferred device if it's supported
            self.active_device = preferred_device
            logger.info(f"Using {preferred_device.name} as requested")
            return preferred_device

        # No preference specified, auto-select based on availability
        if torch.cuda.is_available():
            # CUDA is available, use it
            self.active_device = ComputeDevice.CUDA
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Auto-selected CUDA GPU: {gpu_name}")
        elif self.num_cores > 1:
            # Multiple CPU cores, use multiprocessing
            self.active_device = ComputeDevice.CPU_MULTI
            logger.info(f"Using CPU multiprocessing with {self.num_cores} cores")
        else:
            # Single CPU core, use serial processing
            self.active_device = ComputeDevice.CPU_SERIAL
            logger.info("Using CPU serial processing")

        return self.active_device


# Create a singleton instance
device_manager = DeviceManager()


def get_device_manager():
    """Get the singleton DeviceManager instance."""
    return device_manager


if __name__ == "__main__":
    # Test the device manager
    manager = get_device_manager()
    device = manager.select_device()
    manager.print_device_info()
