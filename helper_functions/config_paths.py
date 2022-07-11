class Paths:
  __conf = {
    # Insert paths/to/local/datasets here
    "sintel_mpi": "",
    "kitti15": ""
  }

  __splits = {
    # Used for dataloading internally
    "sintel_train": "training",
    "sintel_eval": "test",
    "kitti_train": "training",
    "kitti_eval": "testing"
  }

  @staticmethod
  def config(name):
    return Paths.__conf[name]

  @staticmethod
  def splits(name):
    return Paths.__splits[name]

class Conf:
  __conf = {
    # Change the following variables according to your system setup.
    "useCPU": False,  # affects all .to(device) calls
    
    # Set to False, if your installation of spatial-correlation-sampler
    "correlationSamplerOnlyCPU": True  # only used for PWCNet
  }

  @staticmethod
  def config(name):
    return Conf.__conf[name]
