FROM continuumio/anaconda3:4.3.1

MAINTAINER Lab41 <info@lab41.org>

RUN mkdir -p /pelops_root
WORKDIR /pelops_root
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD python3 -m etl.convertCsvToJson