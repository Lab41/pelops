# Turning Chips to Features

1. build the docker containers using make:

```bash
make
```

2. map folders with images and and output directory, and run:

```bash
CHIPDIR1=/folder/with/chips && \
OUTPUTDIR=/folder/for/output && \
docker run -v ${CHIPDIR1}:/pelops_root/INPUT_DIR -v ${OUTPUTDIR}:/pelops_root/OUTPUT_DIR l41-pelops-i2v
```

Note: Docker creates output files owned by root. Grant write privileges to OUTPUT_DIR for the current user and add `-u $(id -u $USER)` to the docker run command above to create output files owned by the current user.

3. Advanced, bring your own model:

```bash
CHIPDIR1=/folder/with/chips && \
OUTPUTDIR=/folder/for/output && \
MODELDIR=/folder/with/models && \
MODELFILE=name_of_model_file && \
WEIGHTFILE=name_of_weight_file && \
LAYERNAME=layername && \
docker run -v ${CHIPDIR1}:/pelops_root/INPUT_DIR -v ${OUTPUTDIR}:/pelops_root/OUTPUT_DIR -v ${MODELDIR}:/pelops_root/MODEL_DIR -e MODEL="/pelops_root/${MODELFILE}" -e WEIGHTS="/pelops_root/${WEIGHTFILE}" -e LAYER="${LAYERNAME}" l41-pelops-i2v
```

Run the Siamese model as follows:

```bash
CHIPDIR1=/folder/with/chips && \
CHIPDIR2=/folder/with/other/chips && \
OUTPUTDIR=/folder/for/output && \
MODELDIR=/folder/with/models && \
MODELFILE=name_of_model_file.json && \
WEIGHTFILE=name_of_weight_file.hdf5 && \
VECTORFILE=name_of_VECTOR_file.json && \
docker run -v ${CHIPDIR1}:/pelops_root/INPUT_DIR1 -v ${CHIPDIR2}:/pelops_root/INPUT_DIR2 -v ${OUTPUTDIR}:/pelops_root/OUTPUT_DIR -v ${MODELDIR}:/pelops_root/MODEL_DIR -e WEIGHTS="/pelops_root/MODEL_DIR/${WEIGHTFILE}" -e MODEL="/pelops_root/MODEL_DIR/${MODELFILE}" -e VECTORS="/pelops_root/INPUT_DIR1/${VECTORFILE}" l41-pelops-siamese
```

Run the Ranker to compare two directories as follows:

```bash
CHIPDIR1=/folder/with/chips && \
CHIPDIR2=/folder/with/other/chips && \
OUTPUTDIR=/folder/for/output && \
MODELDIR=/folder/with/models && \
MODELFILE=name_of_model_file.json && \
WEIGHTFILE=name_of_weight_file.hdf5 && \
LAYERNAME=layername && \
docker run -v ${CHIPDIR1}:/pelops_root/INPUT_DIR1 -v ${CHIPDIR2}:/pelops_root/INPUT_DIR2 -v ${OUTPUTDIR}:/pelops_root/OUTPUT_DIR -v ${MODELDIR}:/pelops_root/MODEL_DIR -e WEIGHTS="/pelops_root/MODEL_DIR/${WEIGHTFILE}" -e MODEL="/pelops_root/MODEL_DIR/${MODELFILE}" -e LAYER="${LAYERNAME}" l41-pelops-ranker
```

Note: Docker creates output files owned by root. Grant write privileges to OUTPUT_DIR for the current user and add `-u $(id -u $USER)` to the docker run commands above to create output files owned by the current user.

Run the CSV to JSON docker conversion operations as follows:

```bash
CSV1=/path/to/file1.csv && \
CSV2=/path/to/file2.csv && \
MODE=product && \
JSON=/path/to/output.json && \
docker run -e pelops_csv_1="${CSV1}" -e pelops_csv_2="${CSV2}" -e pelops_csv_mode=${MODE} -e pelops_json="${JSON}" l41-pelops-c2j
```
