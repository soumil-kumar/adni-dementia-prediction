import argparse
from .common import read_config


def main():
    parser = argparse.ArgumentParser(description='ADNI baseline dementia prediction')
    parser.add_argument('--config', default='config.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    training = sub.add_parser('train')
    training.add_argument('--baseline-only', action='store_true')
    sub.add_parser('evaluate')
    prediction = sub.add_parser('predict')
    prediction.add_argument('--input', required=True)
    sub.add_parser('publish-aggregates')
    sub.add_parser('reproduce')
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == 'prepare':
        from .prepare import prepare
        prepare(cfg)
    elif args.command == 'train':
        from .training import baseline
        baseline(cfg)
        if not args.baseline_only:
            from .search import train
            train(cfg)
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        evaluate(cfg)
    elif args.command == 'reproduce':
        from .reproduce import reproduce
        reproduce(cfg)
    elif args.command == 'publish-aggregates':
        from .evaluation import publish_aggregates
        publish_aggregates(cfg)
    elif args.command == 'predict':
        import json
        from pathlib import Path
        import joblib
        from .artifact import input_frame
        predictor = joblib.load(Path(cfg['local_dir']) / 'model.joblib')
        frame = input_frame(json.loads(Path(args.input).read_text()))
        p = predictor.predict_proba(frame)
        print(json.dumps([dict(risk_probability=float(value),risk_percent=float(value*100),
            threshold=predictor.threshold,predicted_positive=bool(value>=predictor.threshold),
            model_version=predictor.version) for value in p],indent=2))


if __name__ == '__main__':
    main()
