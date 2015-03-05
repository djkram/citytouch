#!/bin/bash

if [[ ! -e /var/eb_log ]]; then
    mkdir /var/eb_log
fi

# starts sercices
echo "Running Citytouch Demo !!!"

# Pico
service pico start
#/data/scripts/pico_server &> /var/log/pico.log &

# Apache2
echo "Starting apache2"
exec /usr/sbin/apache2ctl -DFOREGROUND >> /var/eb_log/apache2.log 2>&1