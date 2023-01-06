import argparse
import yaml
from processor import Processor

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Computer Vision Task Training Processor')

    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp')

    parser.add_argument('-model_saved_name', default='resnet50')

    parser.add_argument(
        '--config',
        default='./config/cifar100/resnet50.yaml')


    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')

    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='the interval for printing messages (#iteration)')

    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='the interval for storing models (#iteration)')

    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=100,
        help='the number of classes')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default='Nothing',
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')

    # scheduler
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', help='type of optimizer')

    # loss
    parser.add_argument('--loss', default='CrossEntropyLoss', help='type of optimizer')
    
    return parser


if __name__ == '__main__':
    parser = get_parser()

    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()

        for k in default_arg.keys():
            if k not in key:
                print(f'Wrong arg : {k}')
                assert(k in key)

        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    processor = Processor(arg)
    processor.start()