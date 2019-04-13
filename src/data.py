import csv
import re
import numpy as np
import random
from skimage import io
import globals


def load_metadata():
    metadata = []
    with open(globals.data_root + 'metadata.csv', 'r', newline='') as metadata_file:
        reader = csv.reader(metadata_file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            metadatum = row[:2] + [int(n) for n in re.findall(r'\d+', row[2])]
            metadata.append(metadatum)
    return metadata


class DataProvider():
    metadata = None
    batch_size = 1  # because I get OOM otherwise

    def __init__(self, metadata, first_category_only=False, get_input_vector=flat_vector):
        self.metadata = metadata
        self.first_category_only = first_category_only
        self.get_input_vector = get_input_vector
        self.downscale_factor = 64
        self.img_x, self.img_y = 5, 14

    def __len__(self):
        return len(self.metadata)

    def get_batch(self):
        x, y = self._get_batch(self.batch_size, self.metadata)
        if x is None or y is None:
            raise ValueError("input x or y is none")
        np.swapaxes(x, 0, 1)
        return x, y

    def upscale(self):
        self.downscale_factor = int(self.downscale_factor / 2)
        self.img_x *= 2
        self.img_y *= 2

    # note to self for the future: optimize the batches to use np arrays from the getgo?
    def _get_batch(self, batch_size, metadata):
        img_x, img_y = self.img_x, self.img_y
        batch_x = np.zeros((batch_size, img_x, img_y, 1), dtype=float)
        batch_y = np.zeros((batch_size, img_x, img_y, 1), dtype=float)
        datum_index = random.randint(0, len(metadata) - 1)

        batch_x = self.get_input_vector(batch_y, None)

        for i in range(batch_size):
            img_scaled = None
            datum_index = random.randint(0, len(metadata) - 1)
            j = 0
            while img_scaled is None:
                metadatum = metadata[(datum_index + j) % len(metadata)]
                latent_category = globals.class_to_id[metadatum[0]]
                if latent_category == globals.num_classes - 1 or self.first_category_only is False:
                    img_scaled = self.get_image(metadata, (datum_index + j) % len(metadata))
                else:
                    img_scaled = None
                j += 1

            # put it in a tensor after downscaling it and padding it
            img_downscaled = downscale_local_mean(img_scaled, (self.downscale_factor, self.downscale_factor, 1))
            # normalize channel values
            if img_downscaled.shape[1] < img_y:
                batch_y[i] = np.pad(img_downscaled, ((0, 0), (0, img_y - img_downscaled.shape[1]), (0, 0)), 'maximum')
            else:
                batch_y[i] = img_downscaled[:, :img_y]

            batch_y[i] /= 255.0
            batch_x = self.get_input_vector(batch_y, latent_category)

        return batch_x.reshape(1, -1), batch_y

    # if outside the tolerance range, return None (it's not a valid datum)
    # if to large, crop from both sides to fit
    # if to small, pad with maximum value (white) to fit
    def scale_to_target(self, image, initial_y, target_y, shrink_tolerance, grow_tolerance):
        if (target_y - initial_y > grow_tolerance or initial_y - target_y > shrink_tolerance):
            return None
        elif (initial_y > target_y):
            return image[:target_y]
        else:  # initial_y <= target_y
            padding = (target_y - initial_y) // 2
            return np.pad(image, ((padding, target_y - initial_y - padding), (0, 0), (0, 0)), 'maximum')

    def get_image(self, metadata, datum_index):
        x, y = 160 * 2, 448 * 2
        metadatum = metadata[datum_index]
        img_raw = io.imread(globals.data_root + 'images/' + metadatum[0] + metadatum[1] + '.png', as_gray=True)
        # if len(img_raw.shape) == 2:
        #    return None
        img_raw = np.expand_dims(img_raw, -1)
        img_scaled = self.scale_to_target(img_raw, int(metadatum[2]), x, 10, 120)
        return img_scaled
