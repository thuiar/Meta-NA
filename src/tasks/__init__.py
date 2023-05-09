import importlib
from .tools import creatNoiseClean
def find_task_using_name(task_name):
    """Import the task "tasks/.py".

    In the file, the class called DatasetNametask() will
    be instantiated. It has to be a subclass of Basetask,
    and it is case-insensitive.
    """
    task_filename = "tasks." + task_name + '_task'
    tasklib = importlib.import_module(task_filename)
    task = None
    target_task_name = task_name + 'task'
    for name, cls in tasklib.__dict__.items():
        if name.lower() == target_task_name.lower():
            task = cls

    if task is None:
        print("In %s.py, there should be a subclass of Basetask with class name that matches %s in lowercase." % (task_filename, target_task_name))
        exit(0)

    return task


def get_task_option_setter(task_name):
    """Return the static method <modify_commandline_options> of the task class."""
    task_class = find_task_using_name(task_name)
    return task_class.modify_commandline_options


def create_task(opt):
    """Create a task given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from tasks import create_task
        >>> task = create_task(opt)
    """
    task = find_task_using_name(opt.task)
    instance = task(opt)
    print("task [%s] was created" % type(instance).__name__)
    return instance

