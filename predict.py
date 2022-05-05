import click
import numpy as np
import pandas as pd

import utils
from network.CNNPredictor import CNNPredictor as Predictor
from network.cnn.training.normalize import StandardScaler


@click.command()
@click.option('--network', '-n', type=click.Path(exists=True), required=True)
@click.option('--state', '-s', 'state', type=click.Path(exists=True), required=True)
@click.option('--project', '-p', type=click.Path(), help='Path to the project directory')
@click.option('--img', 'img_dir', type=click.Path(exists=True))
@click.option('--dest', '-d', type=click.Path())
@click.option('--batch-size', '-b', type=int, default=1)
@click.option('--n-workers', '-nw', type=int, default=4)
def predict(network, state, project, img_dir, dest, batch_size, n_workers):
    img_dir = utils.get_default_path(img_dir, project, 'image')
    dest = utils.get_default_path(dest, project, 'pred_feature.csv', required_exists=False)

    imgs = utils.glob_by_suffixes(img_dir, utils.IMAGE_EXTENSIONS)

    print('Loading predictor...')
    predictor = Predictor(network)
    outputs = predictor.predict(imgs, batch_size=batch_size, n_workers=n_workers)

    # Revert data scaling during training
    print('Saving predicted results...')
    loaded = np.load(state)
    headers = loaded['header']
    scaler = StandardScaler((loaded['mean'], loaded['std']))
    outputs = scaler.inv(outputs)

    # save csv
    indexes = [img.name for img in imgs]
    df = pd.DataFrame(data=outputs, index=indexes, columns=headers)
    df.to_csv(dest, index_label='image')
    

if __name__ == "__main__":
    predict()


    print(.......................)

print("DDDDD")