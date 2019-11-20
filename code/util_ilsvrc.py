"""Read ILSVRC classification labels."""

LABEL_FILE = '../data/imagenet1000_clsidx_to_labels.txt'

def read_labels():

    labels = []
    try:
        with open(LABEL_FILE, 'r', encoding='utf-8') as file:
            for line in file:
                label = line.split(':')[1]
                label = label.rstrip('\n,}').strip(' \'\"')
                labels.append(label)
    except FileNotFoundError:
        print(f'\nERROR: label file not found ({LABEL_FILE})')
        raise
    return labels
