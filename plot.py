import re
import argparse
import matplotlib.pyplot as plt


def parse_log(log, pattern):
    with open(log, 'rb') as log_file:
        for line in log_file:
            match = re.search(pattern, line)
            if match:
                # yield the first group of the pattern;
                # i.e. the one delimited in parenthesis
                # inside the pattern (...)
                yield match.group(1)


def plot_loss(log):
    losses = [float(i) for i in parse_log(log, r'Train loss: (.*)')]
    plt.plot(losses)
    plt.show()


def plot_accuracy(log):
    accuracy = [float(i) for i in parse_log(log, r'.* Accuracy = (\d+\.\d+)%')]
    details = ['exist', 'number', 'material', 'size', 'shape', 'color']
    
    accs = {k: [float(i) for i in parse_log(log, '{} -- acc: (\d+\.\d+)%'.format(k))]
            for k in details}
    
    for k, v in accs.iteritems():
        plt.plot(v, label=k)
    
    plt.plot(accuracy, linewidth=2, label='total')
    plt.legend(loc='best')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot RN training logs')
  parser.add_argument('log_file', type=str, help='Log file to plot')
  parser.add_argument('-l', '--loss', action='store_true', help='Show training loss plot')
  parser.add_argument('-a', '--accuracy', action='store_true', help='Show accuracy plot')
  args = parser.parse_args()
  
  if args.loss:
      plot_loss(args.log_file)
  
  if args.accuracy:
      plot_accuracy(args.log_file)
