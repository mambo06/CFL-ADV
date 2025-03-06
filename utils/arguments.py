# https://github.com/AstraZeneca/SubTab

from argparse import ArgumentParser
import sys
import torch as th
from utils.utils import get_runtime_and_model_config, print_config


class ArgParser(ArgumentParser):
    """Inherits from ArgumentParser, and used to print helpful message if an error occurs"""
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
        
def get_arguments():
    """Gets command line arguments"""
    
    # Initialize parser
    parser = ArgParser()

    # Dataset can be provided via command line
    parser.add_argument("-d", "--dataset", type=str, default="mnist", 
                        help='Name of the dataset to use. It should have a config file with the same name.')
    
    # Whether to use GPU.
    parser.add_argument("-g", "--gpu", dest='gpu', action='store_true', 
                        help='Used to assign GPU as the device, assuming that GPU is available')

    # Whether to use MAC GPU.
    parser.add_argument("-m", "--mps", dest='mps', action='store_true', 
                        help='Used to assign MAC M1 GPU as the device, assuming that GPU is available')
    
    parser.add_argument("-ng", "--no_gpu", dest='gpu', action='store_false', 
                        help='Used to assign CPU as the device')
    
    parser.set_defaults(gpu=True)
    
    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-dn", "--device_number", type=str, default='0', 
                        help='Defines which GPU to use. It is 0 by default')
    
    # Experiment number if MLFlow is on
    parser.add_argument("-ex", "--experiment", type=int, default=1, 
                        help='Used as a suffix to the name of MLFlow experiments if MLFlow is being used')

    ## addition here
    parser.add_argument("-lc", "--local", dest='local', action='store_true')
    parser.add_argument("-e", "--epoch", type=int, default=5,)
    parser.add_argument("-c", "--client", type=int, default=4, )
    parser.add_argument("-cd", "--clientdrop", type=float, default=0, )
    parser.add_argument("-dd", "--datadrop", type=float, default=0, )
    parser.add_argument("-nc", "--noniidclient", type=float, default=0, )
    parser.add_argument("-ci", "--classimbalance", type=float, default=0,)
    parser.add_argument("-b", "--baseGlobal", dest='baseGlobal', action='store_true')
    
    # Return parser arguments
    return parser.parse_args()


def get_config(args):
    """Loads options using yaml files under /config folder and adds command line arguments to it"""
    # Load runtime config from config folder: ./config/ and flatten the runtime config
    config = get_runtime_and_model_config(args)

    # Define which device to use: GPU or CPU

    if th.cuda.is_available() and args.gpu :
        config["device"] = th.device('cuda:' + args.device_number)
    elif th.backends.mps.is_built() and args.mps :
        print("Mac GPU is available ")
        config["device"] = th.device('mps')
    else:
        config["device"] = th.device('cpu')


    config['local'] = args.local
    config['epochs'] = args.epoch
    config['fl_cluster'] = args.client
    config['client_drop_rate'] = args.clientdrop
    config['data_drop_rate'] = args.datadrop
    config['client_imbalance_rate'] = args.noniidclient
    config['class_imbalance'] = args.classimbalance
    config['baseGlobal'] = args.baseGlobal

    # config["device"] = th.device('cuda:' + args.device_number if th.cuda.is_available() and args.gpu else 'cpu')
    # Return
    return config


def print_config_summary(config, args=None):
    """Prints out summary of options and arguments used"""
    # Summarize config on the screen as a sanity check
    print(100 * "=")
    print(f"Here is the configuration being used:\n")
    print_config(config)
    print(100 * "=")
    if args is not None:
        print(f"Arguments being used:\n")
        print_config(args)
        print(100 * "=")
