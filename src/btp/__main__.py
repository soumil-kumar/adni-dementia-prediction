import argparse
from .common import read_config


def main():
    parser = argparse.ArgumentParser(description='ADNI baseline dementia prediction')
    parser.add_argument('--config', default='config.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == 'prepare':
        from .prepare import prepare
        prepare(cfg)


if __name__ == '__main__':
    main()
