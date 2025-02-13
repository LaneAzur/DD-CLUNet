import os

def log_csv(save_dir, *args):
    # check if directory exists
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    with open('logs/'+save_dir+'log.csv', 'a') as f:
        for i, arg in enumerate(args):
            f.write('{}'.format(arg))
            f.write('{}'.format(',' if i != len(args) - 1 else '\n'))
  
def infer_csv(save_dir, csv_name, *args):
    # check if directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir+csv_name, 'a') as f:
        for i, arg in enumerate(args):
            f.write('{}'.format(arg))
            f.write('{}'.format(',' if i != len(args) - 1 else '\n'))
