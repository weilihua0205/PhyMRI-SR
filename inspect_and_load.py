#!/usr/bin/env python3
"""
Inspect a checkpoint's model info without instantiating the model.
Shows parameter shapes, total params, and structure.

Usage:
  python inspect_and_load.py /path/to/ContinuousSR.pth
"""
import sys
import torch
from pprint import pprint
from contextlib import redirect_stdout


def human(n):
    for u in ['','K','M','G']:
        if n < 1000:
            return f"{n:.0f}{u}"
        n /= 1000.0
    return f"{n:.1f}T"


def main():
    if len(sys.argv) < 2:
        print('Usage: python inspect_and_load.py <checkpoint_path>')
        sys.exit(1)

    ckpt_path = sys.argv[1]
    output_file = ckpt_path.replace('.pth', '_inspection.log')
    with open(output_file, 'w') as f:
        with redirect_stdout(f):
            print('Loading:', ckpt_path)
            ck = torch.load(ckpt_path, map_location='cpu')
            print('Type:', type(ck))

            if not isinstance(ck, dict):
                print('Checkpoint is not a dict. repr (truncated):')
                print(repr(ck)[:2000])
                return

            keys = list(ck.keys())
            print('Top-level keys:', keys)

            if 'model' in ck:
                print('\nFound key "model"; summarizing ck["model"]:')
                model = ck['model']
                print('  model type:', type(model))
                if isinstance(model, dict):
                    print('  model keys:', list(model.keys()))
                    if 'name' in model:
                        print('  model["name"] =', model['name'])
                    if 'args' in model:
                        print('  model["args"] keys =', list(model['args'].keys()))
                        print('  model["args"] =', model['args'])
                    if 'sd' in model:
                        sd = model['sd']
                        if isinstance(sd, dict):
                            print('  model["sd"] is a state_dict with %d keys' % len(sd.keys()))
                            total_params = 0
                            print('  Per-parameter summary:')
                            for k, v in sd.items():
                                if hasattr(v, 'shape'):
                                    shape = tuple(v.shape)
                                    num = v.numel()
                                    total_params += num
                                    print(f'    {k}: {shape}, params={human(num)}')
                                else:
                                    print(f'    {k}: {type(v)} (not a tensor)')
                            print('  Total parameters in state_dict:', total_params, '  (~', human(total_params), ')')
                        else:
                            print('  model["sd"] is not a dict; type:', type(sd))
                    else:
                        print('  No "sd" key in model.')
                else:
                    print('  model is not a dict; repr:')
                    print(repr(model)[:2000])
            else:
                print('\nNo "model" key found. If this is a plain state_dict, show #keys and some sample names:')
                if all(isinstance(k, str) for k in keys):
                    print('  Number of keys:', len(keys))
                    total_params = 0
                    print('  sample keys and shapes:')
                    for k in keys[:50]:  # show more
                        v = ck[k]
                        if hasattr(v, 'shape'):
                            shape = tuple(v.shape)
                            num = v.numel()
                            total_params += num
                            print(f'   {k}: {shape}, params={human(num)}')
                        else:
                            print(f'   {k}: {type(v)} (not a tensor)')
                    print('  Total parameters in checkpoint:', total_params, '  (~', human(total_params), ')')
                else:
                    print('  Top-level is a dict but keys are not strings. repr:')
                    print(repr(ck)[:2000])

            print('\nDone')


if __name__ == '__main__':
    main()
