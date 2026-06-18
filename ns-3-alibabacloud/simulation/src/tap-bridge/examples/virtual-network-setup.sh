#!/bin/sh
brctl addbr br-left
brctl addbr br-right
tunctl -t tap-left
tunctl -t tap-right
ifconfig tap-left 0.0.0.0 promisc up
ifconfig tap-right 0.0.0.0 promisc up
brctl addif br-left tap-left
ifconfig br-left up
brctl addif br-right tap-right
ifconfig br-right up
# Note:  you also need to have loaded br_netfilter module
# ('modprobe br_netfilter') to enable /proc/sys/net/bridge
pushd /proc/sys/net/bridge
for f in bridge-nf-*; do echo 0 > $f; done
popd
# lxc-create now requires a template parameter, such as '-t ubuntu'
lxc-create -n left -t ubuntu -f lxc-left.conf
lxc-create -n right -t ubuntu -f lxc-right.conf
# Start both containers
lxc-start -n left -d
lxc-start -n right -d
