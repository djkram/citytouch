#!/bin/bash

# starts sercices
echo "Running Citytouch Demo !!!"
mkdir /var/eb_log

# Pico
service pico start
#/data/scripts/pico_server &> /var/log/pico.log &

# Apache2
echo "Starting apache2"
exec /usr/sbin/apache2ctl -DFOREGROUND >> /var/eb_log/apache2.log 2>&1