import sys
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import re
from struct import unpack
from tiffwrite import IJTiffFile
from tqdm.auto import tqdm
from itertools import product
from yaml import safe_load


class IssTrack:
    def __init__(self, file):
        self.zip = zipfile.ZipFile(file)
        self.data = self.zip.open('data/PrimaryDecayData.bin')
        self.metadata = ET.fromstring(self.zip.read('dataProps/Core.xml'))
        dimensions = self.metadata.find('Dimensions')
        size_t = int(dimensions.find('TimeSeriesCount').text)
        size_c = int(dimensions.find('ChannelCount').text)
        size_y = int(dimensions.find('FrameHeight').text)
        size_x = int(dimensions.find('FrameWidth').text)
        # X, Y, channels, n_images, n_carpets
        self.shape = size_x, size_y, size_c, size_t // 2 + size_t % 2, size_t // 2
        self.exposure_time = float(self.metadata.find('FrameIntervalTime').text)
        self.pxsize = float(self.metadata.find('Boundary').find('FrameWidth').text) / self.shape[0]

        self.alba_metadata = safe_load('\n'.join([IssTrack.parse_line(line)
            for line in self.metadata.find('AlbaSystemSettings').find('withComments').text.splitlines()]))
        particle_tracking = self.alba_metadata['ParticleTracking']
        self.points_per_orbit = particle_tracking['ScanCirclePointCount']
        self.n_orbits = particle_tracking['OrbitCountPerTrack']
        self.orbit_time = particle_tracking['PacketTrackTime_ms']
        self.start_times = [float(re.match(r'[.\d]+', value).group(0))
                            for key, value in self.alba_metadata.items()
                            if re.match(r'Time Series (\d+) started at', key)]
        self.time_interval = np.mean(np.diff(self.start_times)[1::2])
        self.delta = self.shape[2] * 2 ** int(np.ceil(np.log2(self.time_interval * 1000 / self.orbit_time)))
        self.delta = 768  # TODO: figure out if this number is the same for all files

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        try:
            self.data.close()
        finally:
            self.zip.close()

    def get_image(self, c, t):
        assert c < self.shape[2] and t < self.shape[3], \
            f'carpet {c = }, {t = } not in shape {self.shape[2]}, {self.shape[3]}'
        frame = c + 2 * t * self.shape[2]
        frame_bytes = self.shape[0] * self.shape[1] * self.delta
        data = []
        for address in range(frame * frame_bytes, (frame + 1) * frame_bytes, self.delta):
            self.data.seek(address)
            data.append(unpack('<I', self.data.read(4)))
        return np.reshape(data, self.shape[:2])

    def get_carpet(self, c, t):
        assert c < self.shape[2] and t < self.shape[4],\
            f'carpet {c = }, {t = } not in shape {self.shape[2]}, {self.shape[4]}'
        frame = c + (2 * t + 1) * self.shape[2]
        frame_bytes = self.shape[0] * self.shape[1] * self.delta
        data = []
        self.data.seek(frame * frame_bytes)
        for i in range(frame_bytes // (2 * self.points_per_orbit * self.n_orbits)):
            line = [unpack('<H', self.data.read(2))[0] for _ in range(self.points_per_orbit * self.n_orbits)]
            if np.any(line):
                data.append(line)
            else:
                break
        return np.vstack(data)

    def save_images_as_tiff(self, file):
        with IJTiffFile(file, (self.shape[2], 1, self.shape[3]),
                        pxsize=self.pxsize, comment=ET.tostring(self.metadata)) as tif:
            for c, t in tqdm(product(range(self.shape[2]), range(self.shape[3])), total=self.shape[2] * self.shape[3]):
                tif.save(self.get_image(c, t), c, 0, t)

    def save_carpets_as_tiff(self, file):
        with IJTiffFile(file, (self.shape[2], 1, self.shape[4]),
                        pxsize=self.pxsize, comment=ET.tostring(self.metadata)) as tif:
            for c, t in tqdm(product(range(self.shape[2]), range(self.shape[4])), total=self.shape[2] * self.shape[4]):
                tif.save(self.get_carpet(c, t), c, 0, t)

    @staticmethod
    def parse_line(line):
        line = re.sub(r'^(\s*)\[([^\[\]]+)]', r'\1\2:', line)
        line = re.sub(r'\s+:', r':', line)
        line = re.sub(r'\t', r'', line)
        line = re.sub(r'\s*-\s*:', r':', line)
        line = re.sub(r'\s*=\s*', r' ', line)
        return line


def main():
    for file in sys.argv[1:]:
        with IssTrack(file) as iss_file:
            iss_file.save_images_as_tiff(file.replace('.iss-pt', '.tif'))
            iss_file.save_carpets_as_tiff(file.replace('.iss-pt', '.carpet.tif'))


if __name__ == '__main__':
    main()
