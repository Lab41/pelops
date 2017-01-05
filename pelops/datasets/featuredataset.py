import json
import datetime
import h5py
import numpy as np
from pelops.datasets.chip import ChipDataset, Chip

class FeatureDataset(ChipDataset):
    def __init__(self, filename):
        super().__init__(filename)
        self.chip_index_lookup, self.chips, self.feats = self.load(filename)
        self.filename_lookup = {}
        for chip_key, chip in self.chips.items():
            self.filename_lookup[chip.filepath] = chip_key
    
    def get_feats_for_chip(self, chip):
        chip_key = self.filename_lookup[chip.filepath]
        return self.feats[self.chip_index_lookup[chip_key]]
    
    @staticmethod
    def load(filename):
        with h5py.File(filename) as fIn:
            feats = np.array(fIn['feats'])
            
            num_items = fIn['feats'].shape[0]
            # Hack to deal with performance of extracting single items
            local_hdf5 = {}
            local_hdf5['chip_keys'] = np.array(fIn['chip_keys'])
            local_hdf5['filepath'] = np.array(fIn['filepath'])
            local_hdf5['car_id'] = np.array(fIn['car_id'])
            local_hdf5['cam_id'] = np.array(fIn['cam_id'])
            local_hdf5['time'] = np.array(fIn['time'])
            local_hdf5['misc'] = np.array(fIn['misc'])
            
            chips = {}
            chip_index_lookup = {}
            for i in range(num_items):
                filepath = local_hdf5['filepath'][i].decode('utf-8')
                car_id = local_hdf5['car_id'][i]
                cam_id = local_hdf5['cam_id'][i]
                timestamp = local_hdf5['time'][i]
                if isinstance(timestamp, str) or isinstance(timestamp, bytes):
                    # Catch the case where we have encoded time as a string timestamp
                    timestamp = datetime.datetime.fromtimestamp(float(timestamp))
                misc = json.loads(local_hdf5['misc'][i].decode('utf-8'))
                chip_key = local_hdf5['chip_keys'][i]
                if isinstance(chip_key, bytes):
                    chip_key = chip_key.decode('utf-8')
                chip_index_lookup[chip_key] = i
                chips[chip_key] = Chip(filepath, car_id, cam_id, timestamp, misc)
            return chip_index_lookup, chips, feats

    @staticmethod
    def _save_field(fOut, field_example, field_name, value_array):
        if isinstance(field_example, datetime.datetime):
            # Encode time as a string seconds since epoch
            times = np.array([str(val.timestamp()).encode('ascii', 'ignore') for val in value_array])
            fOut.create_dataset(field_name,
                                data=times,
                                dtype=h5py.special_dtype(vlen=bytes))
        elif isinstance(field_example, str):
            output_vals = [val.encode('ascii', 'ignore') for val in value_array]
            fOut.create_dataset(field_name,
                                data= output_vals,
                                dtype=h5py.special_dtype(vlen=bytes))
        elif isinstance(field_example, dict):
            output_vals = [json.dumps(val).encode('ascii', 'ignore') for val in value_array]
            fOut.create_dataset(field_name,
                                data=output_vals,
                                dtype=h5py.special_dtype(vlen=bytes))
        else:
            fOut.create_dataset(field_name, data=value_array)
    
    @staticmethod
    def save(filename, chip_keys, chips, features):
        """ Save a feature dataset
        """
        with h5py.File(filename, 'w') as fOut:
            fOut.create_dataset('feats', data=features)

            FeatureDataset._save_field(fOut,
                                       chip_keys[0],
                                       'chip_keys',
                                       chip_keys)

            first_chip = chips[0]
            fields = first_chip._fields
            for field in fields:
                field_example = getattr(first_chip, field)
                output_data = [getattr(chip, field) for chip in chips]
                FeatureDataset._save_field(fOut, field_example, field, output_data)
