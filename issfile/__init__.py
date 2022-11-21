import re
import pickle
import xml.etree.ElementTree as ET
import numpy as np
from glob import glob
from pathlib import Path
from zipfile import ZipFile
from struct import unpack
from tiffwrite import IJTiffFile, Tag
from tqdm.auto import tqdm
from itertools import product
from yaml import safe_load
from argparse import ArgumentParser
from atexit import register
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ._version import __version__, __git_commit_hash__


class TiffFile(IJTiffFile):
    """ Modify the tiff writer so that it can read from the .iss-pt file by itself in parallel processes. """
    def __init__(self, *args, iss, bar, **kwargs):
        if 'processes' not in kwargs:
            kwargs['processes'] = 'all'
        super().__init__(*args, **kwargs)
        self.iss = pickle.dumps(iss)  # force iss to be transferred to other processes in pickled state
        self.bar = bar

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != 'bar'}

    def update(self):
        self.bar.update()

    def compress_frame(self, frame):
        if isinstance(self.iss, bytes):
            self.iss = pickle.loads(self.iss)
        if frame[0]:
            frame, metadata = self.iss.get_carpet(*frame[1:])
            ifd, offsets = super().compress_frame(frame.astype(self.dtype))
            # ISS = 9*19*19 = 3249; list of t, x, y, z for each row in carpet
            ifd[3249] = Tag('float', metadata.T.flatten())
            return ifd, offsets
        else:
            frame = self.iss.get_image(*frame[1:])
            return super().compress_frame(frame.astype(self.dtype))

    def clean(self):
        try:
            self.iss.close()
        except Exception:
            pass


class IssFile:
    def __init__(self, file, version=388):
        self.file = file
        self.version = version
        self.zip = ZipFile(self.file)
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
        self.alba_metadata = safe_load('\n'.join([IssFile.parse_line(line)
                                                  for line in self.metadata.find('AlbaSystemSettings')
                                                                  .find('withComments').text.splitlines()]))
        particle_tracking = self.alba_metadata['ParticleTracking']
        self.points_per_orbit = particle_tracking['ScanCirclePointCount']
        self.orbits_per_cycle = particle_tracking['OrbitCountPerTrack']
        self.radius = particle_tracking['ScanRadius_um']
        self.cycle_time = particle_tracking['TrackTime_ms']
        self.orbit_pxsize = self.radius * 2 * np.pi / self.points_per_orbit
        self.data_bytes_len = self.zip.getinfo('data/PrimaryDecayData.bin').file_size
        self.delta = self.data_bytes_len // (self.shape[0] * self.shape[1] * self.shape[2] *
                                             (self.shape[3] + self.shape[4]))
        self._carpet_t0 = [0]

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key not in ('zip', 'data')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.zip = ZipFile(self.file)
        self.data = self.zip.open('data/PrimaryDecayData.bin')
        register(self.close)

    def close(self):
        try:
            self.data.close()
        finally:
            self.zip.close()

    def get_image(self, c, t):
        assert c < self.shape[2] and t < self.shape[3], \
            f'carpet {c = }, {t = } not in shape {self.shape[2]}, {self.shape[3]}'
        frame = int(c) + 2 * int(t) * self.shape[2]
        frame_bytes = self.shape[0] * self.shape[1] * self.delta
        data = []
        for address in range(frame * frame_bytes, (frame + 1) * frame_bytes, self.delta):
            self.data.seek(address)
            data.append(unpack('<I', self.data.read(4)))
        return np.reshape(data, self.shape[:2])

    def get_carpet(self, c, t, min_n_lines=1):
        assert c < self.shape[2] and t < self.shape[4], \
            f'carpet {c = }, {t = } not in shape {self.shape[2]}, {self.shape[4]}'
        frame = int(c) + (2 * int(t) + 1) * self.shape[2]
        frame_bytes = self.shape[0] * self.shape[1] * self.delta
        data, metadata = [], []
        self.data.seek(frame * frame_bytes)
        for i in range(frame_bytes // (2 * self.points_per_orbit * self.orbits_per_cycle)):
            line = [unpack('<H', self.data.read(2))[0] for _ in range(self.points_per_orbit * self.orbits_per_cycle)]
            if np.any(line) or i < min_n_lines:
                data.append(line)
                if self.version >= 388:
                    metadata.append(unpack('<ffff', self.data.read(16)))
                else:
                    metadata.append([i * self.cycle_time, 0, 0, 0])
            else:
                break

        data, metadata = np.vstack(data), np.vstack(metadata)
        index = np.zeros(int(round(max(metadata[:, 0]) / self.cycle_time)) + 1, int)
        for i, j in enumerate(metadata[:, 0]):
            index[int(round(j / self.cycle_time))] = i
        metadata[:, 0] += self.get_carpet_t0(t, metadata.shape[0])
        return data[index], metadata[index].T

    def get_carpet_t0(self, t, cycles=None):
        if t + 1 > len(self._carpet_t0):
            cycles = cycles or self.get_carpet(0, t - 1)[1].shape[1]
            t0 = self.get_carpet_t0(t - 1)
            self._carpet_t0.append(t0 + cycles * self.cycle_time + self.exposure_time)
        return self._carpet_t0[t]

    def save_images_as_tiff(self, file):
        if self.shape[3]:
            with tqdm(total=self.shape[2] * self.shape[3], desc='Writing  images') as bar:
                with TiffFile(file, (self.shape[2], 1, self.shape[3]), iss=self, bar=bar,
                              pxsize=self.pxsize, comment=ET.tostring(self.metadata)) as tif:
                    for c, t in product(range(self.shape[2]), range(self.shape[3])):
                        tif.save((0, c, t), c, 0, t)

    def save_carpets_as_tiff(self, file):
        if self.shape[4]:
            with tqdm(total=self.shape[2] * self.shape[4], desc='Writing carpets') as bar:
                with TiffFile(file, (self.shape[2], 1, self.shape[4]), iss=self, bar=bar,
                              pxsize=self.orbit_pxsize, comment=ET.tostring(self.metadata)) as tif:
                    for c, t in product(range(self.shape[2]), range(self.shape[4])):
                        tif.save((1, c, t), c, 0, t)

    @staticmethod
    def parse_line(line):
        line = re.sub(r'^(\s*)\[([^\[\]]+)]', r'\1\2:', line)
        line = re.sub(r'\s+:', r':', line)
        line = re.sub(r'\t', r'', line)
        line = re.sub(r'\s*-\s*:', r':', line)
        line = re.sub(r'\s*=\s*', r' ', line)
        return line

    def get_txyzi(self, c):
        data, metadata = zip(*[self.get_carpet(c, t) for t in range(self.shape[4])])
        txyz = np.hstack(metadata)
        idx = np.all(txyz[1:] != 0, 0)
        return *txyz[:, idx], np.hstack([d.sum(1) for d in data])[idx]

    def save_plot(self, file):
        if self.shape[4]:
            with PdfPages(file) as pdf:
                plt.figure(figsize=(8.3, 11.7))
                plt.subplot(3, 1, 1)
                for c, color in zip(range(self.shape[2]), 'rgbm'):
                    t, x, y, z, intensity = self.get_txyzi(c)
                    plt.plot(t / 1000, intensity, color)
                plt.xlabel('time (s)')
                plt.ylabel('sum intensity (counts)')
                plt.legend([f'channel {i}' for i in range(self.shape[2])])
                plt.subplot(3, 1, 2)
                for i, color in zip((x, y, z), 'rgb'):
                    plt.plot(t / 1000, i, color)
                plt.xlabel('time (s)')
                plt.ylabel('position (μm)')
                plt.legend('xyz')
                plt.subplot(3, 3, 7)
                plt.plot(x, y, 'k')
                plt.axis('equal')
                plt.xlabel('x (μm)')
                plt.ylabel('y (μm)')
                plt.subplot(3, 3, 8)
                plt.plot(x, z, 'k')
                plt.axis('equal')
                plt.xlabel('x (μm)')
                plt.ylabel('z (μm)')
                plt.subplot(3, 3, 9)
                plt.plot(y, z, 'k')
                plt.axis('equal')
                plt.xlabel('y (μm)')
                plt.ylabel('z (μm)')
                plt.tight_layout()
                pdf.savefig()


def main():
    parser = ArgumentParser(description='Convert .iss-pt files into .tiff files.')
    parser.add_argument('files', help='files to be converted', nargs='*')
    parser.add_argument('-v', '--version', type=int, default=388,
                        help='version of VistaVision with which the .iss-pt was written, default: 388')
    args = parser.parse_args()

    for file in [Path(file) for files in args.files for file in glob(files)]:
        with IssFile(file, args.version) as iss_file:
            iss_file.save_images_as_tiff(file.with_suffix('.tif'))
            iss_file.save_carpets_as_tiff(file.with_suffix('.carpet.tif'))
            iss_file.save_plot(file.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
