import abc
import json
import datetime
import h5py
import numpy as np
from pelops.datasets.chip import ChipDataset, Chip

class FeatureDataset(ChipDataset):
    def __init__(self, filename):
        # TODO: Call super
        super().__init__(filename)
        self.chips, self.feats = self.load(filename)
        self.filename_lookup = {}
        for chip_key, chip in self.chips.items():
            self.filename_lookup[chip.filepath] = chip_key
    
    def get_feats_for_chip(self, chip):
        return self.feats[self.filename_lookup[chip.filepath]]
    
    @staticmethod
    def load(filename):
        with h5py.File(filename) as fIn:
            feats = np.array(fIn['feats'])
            
            num_items = fIn['feats'].shape[0]
            # Hack to deal with performance of extracting single items
            local_hdf5 = {}
            local_hdf5['filepath'] = np.array(fIn['filepath'])
            local_hdf5['car_id'] = np.array(fIn['car_id'])
            local_hdf5['cam_id'] = np.array(fIn['cam_id'])
            local_hdf5['time'] = np.array(fIn['time'])
            local_hdf5['misc'] = np.array(fIn['misc'])
            
            chips = {}
            for i in range(num_items):
                filepath = local_hdf5['filepath'][i].decode('utf-8')
                car_id = local_hdf5['car_id'][i]
                cam_id = local_hdf5['cam_id'][i]
                time = datetime.datetime.fromtimestamp(local_hdf5['time'][i]/1000.0)
                misc = json.loads(local_hdf5['misc'][i].decode('utf-8'))
                chips[i] = Chip(filepath, car_id, cam_id, time, misc)
            return chips, feats
    
    @staticmethod
    def save(self, filename, chips, features):
        with h5py.File(filename, 'w') as fOut:
            fOut.create_dataset('feats', data=features)
            for field in chips[0]._fields:
                if isinstance(getattr(chips[0], field), datetime.datetime):
                    times = np.array([getattr(chip, field).timestamp() for chip in chips])
                    times = times * 1000.0 # Convert to ms since epoch
                    fOut.create_dataset(field, data=times, dtype=np.int64)
                elif isinstance(getattr(chips[0], field), str):
                    fOut.create_dataset(field, 
                                        data=[getattr(chip, field).encode('ascii', 'ignore') for chip in chips],
                                        dtype=h5py.special_dtype(vlen=bytes))
                elif isinstance(getattr(chips[0], field), dict):
                    data = [json.dumps(getattr(chip, field)).encode('ascii', 'ignore') for chip in chips]
                    fOut.create_dataset(field, 
                                        data=data,
                                        dtype=h5py.special_dtype(vlen=bytes))
                else:
                    fOut.create_dataset(field, data=[getattr(chip, field) for chip in chips])
