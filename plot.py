import re
import argparse
import os
import matplotlib

def parse_log(log, pattern):
    with open(log, 'r') as log_file:
        for line in log_file:
            match = re.search(pattern, line)
            if match:
                # yield the first group of the pattern;
                # i.e. the one delimited in parenthesis
                # inside the pattern (...)
                yield match.group(1)

def plot_train_loss(args):
    losses = [float(i) for i in parse_log(args.log_file, r'Train loss: (.*)')]
    plt.clf()
    plt.plot(losses)
    plt.title('Train Loss')
    plt.xlabel('Iteration')
    if args.y_max != 0:
        plt.ylim(ymax=args.y_max)
    if args.y_min != 0:
        plt.ylim(ymin=args.y_min)
    plt.savefig(os.path.join(args.img_dir, 'train_loss.png'))
    if not args.no_show:
        plt.show()

def plot_test_loss(args):
    losses = [float(i) for i in parse_log(args.log_file, r'Test loss = (.*)')]
    plt.clf()
    plt.plot(losses)
    plt.title('Test Loss')
    if args.y_max != 0:
        plt.ylim(ymax=args.y_max)
    if args.y_min != 0:
        plt.ylim(ymin=args.y_min)
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(args.img_dir, 'test_loss.png'))
    if not args.no_show:
        plt.show()


def plot_accuracy(args):
    accuracy = [float(i) for i in parse_log(args.log_file, r'.* Accuracy = (\d+\.\d+)%')]
    details = ['exist', 'number', 'material', 'size', 'shape', 'color']
    
    accs = {k: [float(i) for i in parse_log(args.log_file, '{} -- acc: (\d+\.\d+)%'.format(k))]
            for k in details}
    
    plt.clf()
    for k, v in accs.items():
        plt.plot(v, label=k)

    plt.plot(accuracy, linewidth=2, label='total')
    plt.legend(loc='best')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.savefig(os.path.join(args.img_dir, 'accuracy.png'))
    if not args.no_show:
        plt.show()

def plot_invalids(args):
    invalids = [float(i) for i in parse_log(args.log_file, r'.* Invalids = (\d+\.\d+)%')]
    '''details = ['exist', 'number', 'material', 'size', 'shape', 'color']
    
    invds = {k: [float(i) for i in parse_log(log, '.* invalid: (\d+\.\d+)%'.format(k))]
            for k in details}
    
    for k, v in invds.items():
        plt.plot(v, label=k)'''
    
    plt.clf()
    plt.plot(invalids, linewidth=2, label='total')
    plt.legend(loc='best')
    plt.title('Invalid rate')
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.savefig(os.path.join(args.img_dir, 'invalids.png'))
    if not args.no_show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot RN training logs')
    parser.add_argument('log_file', type=str, help='Log file to plot')
    parser.add_argument('-trl', '--train-loss', action='store_true', help='Show training loss plot')
    parser.add_argument('-tsl', '--test-loss', action='store_true', help='Show test loss plot')
    parser.add_argument('-a', '--accuracy', action='store_true', help='Show accuracy plot')
    parser.add_argument('-i', '--invalids', action='store_true', help='Show invalid rate plot')
    parser.add_argument('--no-show', action='store_true', help='Do not show figures, store only on file')
    parser.add_argument('--y-max', type=float, default=0,
                        help='upper bound for y axis of loss plots (0 to leave default)')
    parser.add_argument('--y-min', type=float, default=0,
                        help='lower bound for y axis of loss plots (0 to leave default)')
    args = parser.parse_args()
    
    img_dir = 'imgs/'
    args.img_dir = img_dir

    if args.no_show:
        matplotlib.use('Agg')    
    import matplotlib.pyplot as plt

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if args.train_loss:
      plot_train_loss(args)

    if args.test_loss:
      plot_test_loss(args)

    if args.accuracy:
      plot_accuracy(args)

    if args.invalids:
      plot_invalids(args)
