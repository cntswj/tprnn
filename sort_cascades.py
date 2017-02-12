import os

data_dir = 'data/twitter'
dataset = 'test'
filepath = os.path.join(data_dir, dataset + '.txt')
output = os.path.join(data_dir, dataset + '_sorted.txt')

with open(filepath, 'rb') as f, open(output, 'wb') as fo:
    for line in f:
        query, cascade = line.strip().split(' ', 1)
        timestamps = map(int, cascade.split(' ')[1::2])
        sequence = cascade.split(' ')[::2]

        # sort by timestamps and keep unique nodes.
        ordered = sorted(zip(sequence, timestamps), key=lambda x: x[1])
        so_far = set()
        ordered_unique = []
        for v, t in ordered:
            if v not in so_far:
                ordered_unique.append((v, t))
                so_far.add(v)
        if len(ordered_unique) < 2:
            continue

        fo.write(query + ' ')
        fo.write(' '.join(['%s %d' % (v, t) for v, t in ordered_unique]))
        fo.write('\n')
