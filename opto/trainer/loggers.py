

class BaseLogger:

    def log(self, name, data, step, **kwargs):
        """Log a message with the given name and data at the specified step.
        
        Args:
            name: Name of the metric
            data: Value of the metric
            step: Current step/iteration
            **kwargs: Additional arguments (e.g., color)
        """
        raise NotImplementedError("Subclasses should implement this method.")


class ConsoleLogger(BaseLogger):
    """A simple logger that prints messages to the console."""
    
    def log(self, name, data, step, **kwargs):
        """Log a message to the console.
        
        Args:
            name: Name of the metric
            data: Value of the metric
            step: Current step/iteration
            **kwargs: Additional arguments (e.g., color)
        """
        color = kwargs.get('color', None)
        # Simple color formatting for terminal output
        color_codes = {
            'green': '\033[92m',
            'red': '\033[91m',
            'blue': '\033[94m',
            'end': '\033[0m'
        }
        
        start_color = color_codes.get(color, '')
        end_color = color_codes['end'] if color in color_codes else ''
        
        print(f"[Step {step}] {start_color}{name}: {data}{end_color}")


class TensorboardLogger(BaseLogger):
    """A logger that writes metrics to TensorBoard."""
    
    def __init__(self, log_dir):
        # Late import to avoid dependency issues
        try:             
            from tensorboardX import SummaryWriter
        except ImportError:
            # try importing from torch.utils.tensorboard if tensorboardX is not available
            from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir)

    def log(self, name, data, step, **kwargs):
        """Log a message to TensorBoard.
        
        Args:
            name: Name of the metric
            data: Value of the metric
            step: Current step/iteration
            **kwargs: Additional arguments (not used here)
        """
        self.writer.add_scalar(name, data, step)

# TODO add wandb logger

DefaultLogger = ConsoleLogger