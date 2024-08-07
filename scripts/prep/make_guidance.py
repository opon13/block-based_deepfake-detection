import os, random, csv
import argparse

from dfx import get_path


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir', '--datasets_dir', type=str, default=None)
    parser.add_argument('-guide_dir', '--guidance_dir', type=str, default=None)
    parser.add_argument('-save_dir', '--saving_dir', type=str, default=None)

    args = parser.parse_args()
    return args


def main(parser):

    datasets_path = get_path('dataset') if parser.datasets_dir is not None else parser.datasets_dir
    guidance_path = get_path('guidance') if parser.guidance_dir is not None else parser.guidance_dir
    models_dir = get_path('models') if parser.saving_dir is not None else parser.saving_dir

    for folder in ['bm-dm', 'bm-gan', 'bm-real', 'complete']:
        folder_dir = os.path.join(models_dir, folder)
        if not os.path.exists(folder_dir):
            os.mkdir(folder_dir)

    with open(guidance_path[:-4]+'.txt', 'w') as f:
        for models_name in os.listdir(datasets_path):
            models_path = os.path.join(datasets_path, models_name)
            for architecture_name in os.listdir(models_path):
                architecture_path = os.path.join(models_path, architecture_name)
                for image in os.listdir(architecture_path):
                    image_path = os.path.join(architecture_path, image)
                    x = random.random()
                    label = int(x >= 0.4) + int(x >= 0.8)
                    f.write(f"{label} & {image_path} & {models_name}\n")

    with open(guidance_path[:-4]+'.txt', 'r') as file:
        data = []
        for line in file:
            label, image, model = line.strip().split(' & ')
            data.append((label, image, model))

    with open(guidance_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['label', 'image_path', 'model']) 
        writer.writerows(data)


if __name__=='__main__':
    parser = get_parser()
    print('writing new guidance.csv...')
    main(parser)