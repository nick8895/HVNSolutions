#!/bin/bash
render=base
tensorflow_version="2.11.0"
uid=$(eval "id -u")
gid=$(eval "id -g")

docker build \
  --build-arg TENSORFLOW_VERSION="$tensorflow_version" \
  --build-arg RENDER="$render" \
  --build-arg UID="$uid" \
  --build-arg GID="$gid" \
  -f HVN.Dockerfile \
  -t rp2024/hvn:"$tensorflow_version" .
