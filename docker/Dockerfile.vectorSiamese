FROM l41-pelops-base

MAINTAINER Lab41 <info@lab41.org>

RUN mkdir INPUT_DIR1
RUN mkdir INPUT_DIR2
RUN mkdir MODEL_DIR
RUN mkdir OUTPUT_DIR

CMD ["python", "/pelops_root/etl/makeFeaturesTopSiamese.py","./INPUT_DIR1","./INPUT_DIR2","./OUTPUT_DIR"]
