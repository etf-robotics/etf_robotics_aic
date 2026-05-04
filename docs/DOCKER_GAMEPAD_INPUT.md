# Docker Gamepad Input Notes

These notes describe the changes made to `docker/docker-compose.yaml` so Isaac Sim can receive gamepad input from inside the container.

## Changes

### Bind the full input device tree

Changed the device mapping from only:

```yaml
- /dev/input/js0:/dev/input/js0
```

to:

```yaml
- /dev/input:/dev/input
```

Reason: Linux exposes gamepads through both legacy joystick nodes such as `js0` and event nodes such as `event*`. Omniverse/Carb input discovery often needs the `event*` devices, not only `js0`.

### Bind udev metadata

Added:

```yaml
- type: bind
  source: /run/udev
  target: /run/udev
  read_only: true
```

Reason: Omniverse/Carb may use udev metadata to identify input devices correctly. The container can see `/dev/input/js0` without this, but Isaac may still fail to classify or receive gamepad events.

### Allow Linux input device access

Added to the Isaac Lab services:

```yaml
device_cgroup_rules:
  - "c 13:* rmw"
```

Reason: Linux input devices use character device major number `13`. This cgroup rule allows the container to read/write/mknod those input devices exposed under `/dev/input`.

## After Changing Compose

Recreate the container so the new device mounts and cgroup rules apply:

```bash
cd /home/etfrobot/IsaacLab/docker
docker compose --profile base down
docker compose --profile base up -d
```

Inside the container, verify that `event*` devices and udev metadata are visible:

```bash
ls -l /dev/input
ls -l /run/udev
```

