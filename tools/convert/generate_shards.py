import json
import argparse
import os
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',  type=str)
    parser.add_argument('--out-dir',  type=str)
    parser.add_argument('--num-shards', type=int) 

    args = parser.parse_args()
    data = json.load(open(args.json, 'r'))
    os.makedirs(args.out_dir, exist_ok=True)
    for i in tqdm.tqdm(range(args.num_shards)):
        json.dump(data[i::args.num_shards], open(os.path.join(args.out_dir, f"shards#{args.num_shards}_{i}.json"), 'w'))


if __name__ == '__main__':
    main()
    