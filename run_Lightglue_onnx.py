# -*- coding: utf-8 -*-

import os
import time
import argparse
import onnxruntime as ort

def main():
    parser = argparse.ArgumentParser(description='Run LightGlue demo.')
    parser.add_argument('--inputdir0', type=str , help='xxxx')
    parser.add_argument('--inputdir1', type=str , help='xxxx')
    parser.add_argument('--savedir', type=str , help='xxxx')
    parser.add_argument('--withline', type=str , help='xxxx')

    args = parser.parse_args()
    
    input_dir0 = args.inputdir0 if args.inputdir0 is not None else r"data/dir0"
    input_dir1 = args.inputdir1 if args.inputdir1 is not None else r"data/dir1"
    save_dir = args.savedir if args.savedir is not None else r"data/output"
    withline = args.withline

if __name__ == '__main__':
    main()