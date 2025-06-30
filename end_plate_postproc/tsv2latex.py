import os
import pandas as pd


datasets = ['inner', 'easy', 'hard']
models = ['effunet', 'unet', 'unet3d', 'vnet', 'unetr', 'swinunetr']
keys = ['glob_dice', 'glob_jacc', 'dice', 'jc', 'hd95', 'asd']
for dataset in datasets:
    print('-' * 100)
    print(dataset)
    for model in models:
        table = pd.read_csv(f'scores_{dataset}/{model}.tsv', sep='\t')
        scores = [model]
        if model == 'unet':
            print(len(table))
        #
        prec = table['Prec'].mean()
        reca = table['Reca'].mean()
        f1 = 2 * prec * reca / (prec + reca)
        scores.append(f'{prec:.3f}')
        scores.append(f'{reca:.3f}')
        scores.append(f'{f1:.3f}')
        for key in keys:
            scores.append(f'{table[key].mean():.3f}')
        print(' & '.join(scores) + ' \\\\')

