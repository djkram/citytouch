#!/bin/bash

# starts sercices
echo "Running Citytouch Demo !!!"

# Pico
service pico start
#/data/scripts/pico_server &> /var/log/pico.log &

# Apache2
echo "Starting apache2"
exec /usr/sbin/apache2ctl -DFOREGROUND >> /var/log/apache2.log 2>&1