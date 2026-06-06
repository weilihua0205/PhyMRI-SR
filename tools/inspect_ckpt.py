import sys
import torch
from pprint import pprint

if len(sys.argv) < 2:
    print('Usage: python inspect_ckpt.py <checkpoint_path>')
    sys.exit(1)

path = sys.argv[1]
print('Loading:', path)
ck = torch.load(path, map_location='cpu')
print('Type:', type(ck))

if isinstance(ck, dict):
    keys = list(ck.keys())
    print('Top-level keys:', keys)
    # if it's a plain state_dict (weights only), print some info
    if 'model' in ck:
        print('\nFound key "model"; summarizing ck["model"]:')
        try:
            model = ck['model']
            print('  model type:', type(model))
            if isinstance(model, dict):
                print('  model keys:', list(model.keys()))
                if 'name' in model:
                    print('  model["name"] =', model['name'])
                if 'args' in model:
                    print('  model["args"] keys =', list(model['args'].keys()))
                if 'sd' in model:
                    sd = model['sd']
                    if isinstance(sd, dict):
                        print('  model["sd"] is a state_dict with %d keys' % len(sd.keys()))
                        # print a short sample
                        sample = list(sd.items())[:5]
                        print('  sample param keys:')
                        for k, v in sample:
                            print('    ', k, ',', tuple(v.shape) if hasattr(v, 'shape') else type(v))
            else:
                print('  model is not a dict; repr:')
                print(repr(model)[:2000])
        except Exception as e:
            print('  Error summarizing model:', e)
    else:
        # maybe checkpoint is just a state_dict
        print('\nNo "model" key found. If this is a plain state_dict, show #keys and some sample names:')
        try:
            if all(isinstance(k, str) for k in keys):
                print('  Number of keys:', len(keys))
                print('  sample keys:')
                for k in keys[:30]:
                    print('   ', k)
            else:
                print('  Top-level is a dict but keys are not strings. repr:')
                print(repr(ck)[:2000])
        except Exception as e:
            print('  Error listing keys:', e)
else:
    print('Checkpoint is not a dict. repr (truncated):')
    print(repr(ck)[:2000])

print('\nDone')
