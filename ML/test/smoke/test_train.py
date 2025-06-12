from runners import TrainRunner


def test_train(output_dir, config_path):
    TrainRunner(
        config_path=config_path,
        distributed=False,
        options={
            'output_dir': output_dir / 'train',
            'trainer_params.epochs': 1,
            'model.backbone.pretrained_weights': None,
        },
    )()
