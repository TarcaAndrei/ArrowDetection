from runners import EvalRunner


def test_eval(output_dir, config_path):
    EvalRunner(
        config_path=config_path,
        distributed=False,
        options={
            'output_dir': output_dir / 'eval',
            'model.backbone.pretrained_weights': None,
        },
    )()
