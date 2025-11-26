import torch
import sys
p = r'c:\Users\khotc\Downloads\skin\checkpoints\Swin_MC_best_model.pth'
try:
    ck = torch.load(p, map_location='cpu')
    print('OK: loaded checkpoint, type=', type(ck))
    if isinstance(ck, dict):
        keys = list(ck.keys())
        print('Top-level keys (count=%d):' % len(keys))
        for k in keys[:200]:
            v = ck[k]
            print('-', k, '->', type(v))
        # look for likely mapping keys
        for candidate in ['class_to_idx','classes','label_map','idx_to_class','mapping','meta']:
            if candidate in ck:
                print('\nFound candidate mapping key:', candidate)
                print(ck[candidate])
    else:
        print('Checkpoint is not a dict; likely a raw state_dict mapping of tensors.')
except Exception as e:
    print('ERROR loading checkpoint:', e)
    sys.exit(1)
