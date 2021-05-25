#!/bin/bash
( cd data-service ; exec python3 data_service.py & )
( cd webapp; exec python3 -m flask run --host 0.0.0.0 )
