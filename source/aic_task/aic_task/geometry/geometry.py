"""Central geometry constants for the NIC-card port insertion task.

This module is intentionally standalone for now.  The existing vision,
controller, reward, and termination code should keep using their current
constants until we explicitly migrate them to this geometry spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Final


# Units: meters unless a suffix says otherwise.
Vector3 = tuple[float, float, float]

NIC_CARD_USD_RELATIVE_PATH: Final[str] = "assets/NIC Card/nic_card.usd"

# Paths are relative to the NIC-card asset root.  At runtime they will sit under
# the scene prim for the rigid object, e.g. ``.../nic_card/nic_card_link/...``.
NIC_CARD_LINK_PATH: Final[str] = "/nic_card_link"
VISUAL_PATH: Final[str] = "/nic_card_link/visual"

# Existing USD node on the port edge opposite the keying tooth.  It is a visible
# semantic anchor for port 0; port 1 derives the same anchor by translating this
# offset from port 0's entrance to port 1's entrance.
OPPOSITE_TOOTH_NODE_PATH: Final[str] = "/nic_card_link/visual/node_0099100_011LFC_002"
OPPOSITE_TOOTH_NODE_OLD_POS_NIC: Final[Vector3] = (-0.01025, -0.07725, 0.0103515)
OPPOSITE_TOOTH_NODE_OLD_RPY_DEG_NIC: Final[Vector3] = (179.227, 0.0, 0.0)


@dataclass(frozen=True)
class PortFramePaths:
    """USD frames that define one physical SFP port."""

    name: str
    entrance_path: str
    seat_path: str
    opposite_tooth_node_path: str | None = None
    opposite_tooth_source_port: str | None = None


PORTS: Final[tuple[PortFramePaths, ...]] = (
    PortFramePaths(
        name="sfp_port_0",
        entrance_path="/nic_card_link/sfp_port_0_link/sfp_port_0_link_entrance",
        seat_path="/nic_card_link/sfp_port_0_link",
        opposite_tooth_node_path=OPPOSITE_TOOTH_NODE_PATH,
    ),
    PortFramePaths(
        name="sfp_port_1",
        entrance_path="/nic_card_link/sfp_port_1_link/sfp_port_1_link_entrance",
        seat_path="/nic_card_link/sfp_port_1_link",
        opposite_tooth_source_port="sfp_port_0",
    ),
)
PORTS_BY_NAME: Final[dict[str, PortFramePaths]] = {port.name: port for port in PORTS}

# Port mouth dimensions.  The long half-width is a fixed CAD dimension.  The
# tooth/opposite-tooth half-width should be measured from the composed transform
# between an entrance frame and the opposite-tooth anchor.  The fallback is only
# the nominal value while we are offline.
PORT_LONG_HALF: Final[float] = 0.007
PORT_LONG_TOTAL: Final[float] = 2.0 * PORT_LONG_HALF
PORT_WIDTH_FALLBACK: Final[float] = 0.009
PORT_Y_HALF_FALLBACK: Final[float] = 0.5 * PORT_WIDTH_FALLBACK

# Fallback only.  The source of truth is the composed transform difference
# between ``sfp_port_*_link_entrance`` and ``sfp_port_*_link``.
PLUG_INSERTION_DEPTH_FALLBACK: Final[float] = 0.044

# Semantic local frame at the port entrance:
#   origin: ``sfp_port_*_link_entrance``
#   +X: long dimension of the port mouth
#   -Y: tooth/keying side of the port
#   +Y: opposite-tooth side of the port
#   insertion axis: vector from entrance frame to ``sfp_port_*_link``
LONG_AXIS_LOCAL: Final[Vector3] = (1.0, 0.0, 0.0)
TOOTH_DIRECTION_LOCAL: Final[Vector3] = (0.0, -1.0, 0.0)
OPPOSITE_TOOTH_DIRECTION_LOCAL: Final[Vector3] = (0.0, 1.0, 0.0)


@dataclass(frozen=True)
class PortRuntimePoints:
    """Composed USD points for one port, expressed in a shared parent frame."""

    entrance: Vector3
    seat: Vector3
    opposite_tooth_anchor: Vector3


def vector_sub(a: Vector3, b: Vector3) -> Vector3:
    """Return ``a - b``."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_add(a: Vector3, b: Vector3) -> Vector3:
    """Return ``a + b``."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vector_norm(value: Vector3) -> float:
    """Return Euclidean vector length."""
    return sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def insertion_vector(entrance: Vector3, seat: Vector3) -> Vector3:
    """Return the insertion displacement from entrance center to seated center."""
    return vector_sub(seat, entrance)


def insertion_depth(entrance: Vector3, seat: Vector3) -> float:
    """Compute insertion depth from composed USD entrance and seat positions."""
    return vector_norm(insertion_vector(entrance, seat))


def opposite_tooth_offset(port_0_entrance: Vector3, port_0_opposite_tooth_anchor: Vector3) -> Vector3:
    """Return the offset from port 0 entrance center to its opposite-tooth anchor."""
    return vector_sub(port_0_opposite_tooth_anchor, port_0_entrance)


def derived_opposite_tooth_anchor(
    port_entrance: Vector3,
    *,
    source_port_entrance: Vector3,
    source_opposite_tooth_anchor: Vector3,
) -> Vector3:
    """Translate port 0's opposite-tooth offset onto another translated port."""
    return vector_add(port_entrance, opposite_tooth_offset(source_port_entrance, source_opposite_tooth_anchor))


def port_y_half_from_opposite_anchor(entrance: Vector3, opposite_tooth_anchor: Vector3) -> float:
    """Compute the tooth/opposite-tooth half-width from USD points."""
    return vector_norm(opposite_tooth_offset(entrance, opposite_tooth_anchor))


def make_port_keypoints_local(
    *,
    y_half: float = PORT_Y_HALF_FALLBACK,
    insertion_depth_value: float | None = None,
) -> dict[str, Vector3]:
    """Return entrance-centered local keypoints for a port mouth."""
    keypoints = {
        "entrance_center": (0.0, 0.0, 0.0),
        "opposite_tooth_anchor": (0.0, y_half, 0.0),
        "tooth_anchor": (0.0, -y_half, 0.0),
        "corner_opposite_left": (-PORT_LONG_HALF, y_half, 0.0),
        "corner_opposite_right": (PORT_LONG_HALF, y_half, 0.0),
        "corner_tooth_left": (-PORT_LONG_HALF, -y_half, 0.0),
        "corner_tooth_right": (PORT_LONG_HALF, -y_half, 0.0),
    }
    if insertion_depth_value is not None:
        keypoints["seat_center"] = (0.0, 0.0, insertion_depth_value)
    return keypoints


PORT_KEYPOINTS_LOCAL_FALLBACK: Final[dict[str, Vector3]] = make_port_keypoints_local(
    y_half=PORT_Y_HALF_FALLBACK,
    insertion_depth_value=PLUG_INSERTION_DEPTH_FALLBACK,
)

ROBOT_PLUG_FRAME_NOTES: Final[tuple[str, ...]] = (
    "Use the new robot plug-center and plug-tip frames once their exact USD/body names are confirmed.",
    "Plug orientation should be derived from center -> tip, then aligned to the port insertion vector.",
)


def validate_port_name(name: str) -> PortFramePaths:
    """Return a port spec or raise a helpful error."""
    try:
        return PORTS_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(PORTS_BY_NAME)
        raise KeyError(f"Unknown port '{name}'. Available ports: {available}") from exc


def derived_port_runtime_points(
    port_name: str,
    *,
    entrance: Vector3,
    seat: Vector3,
    port_0_entrance: Vector3,
    port_0_opposite_tooth_anchor: Vector3,
) -> PortRuntimePoints:
    """Build runtime points for a port whose opposite anchor may be translated from port 0."""
    port = validate_port_name(port_name)
    if port.name == "sfp_port_0":
        opposite_anchor = port_0_opposite_tooth_anchor
    elif port.opposite_tooth_source_port == "sfp_port_0":
        opposite_anchor = derived_opposite_tooth_anchor(
            entrance,
            source_port_entrance=port_0_entrance,
            source_opposite_tooth_anchor=port_0_opposite_tooth_anchor,
        )
    else:
        raise ValueError(f"Port '{port.name}' has no opposite-tooth anchor rule.")
    return PortRuntimePoints(entrance=entrance, seat=seat, opposite_tooth_anchor=opposite_anchor)


def keypoints_for_runtime_points(points: PortRuntimePoints) -> dict[str, Vector3]:
    """Return local keypoints using USD-derived width and insertion depth."""
    return make_port_keypoints_local(
        y_half=port_y_half_from_opposite_anchor(points.entrance, points.opposite_tooth_anchor),
        insertion_depth_value=insertion_depth(points.entrance, points.seat),
    )


def print_geometry() -> None:
    """Print the current geometry constants in a readable form."""
    print("NIC-card port geometry")
    print(f"  nic card usd relative path: {NIC_CARD_USD_RELATIVE_PATH}")
    print(f"  nic card link path: {NIC_CARD_LINK_PATH}")
    print(f"  opposite tooth node path: {OPPOSITE_TOOTH_NODE_PATH}")
    print(f"  old opposite tooth node pos in nic_card frame: {OPPOSITE_TOOTH_NODE_OLD_POS_NIC}")
    print(f"  old opposite tooth node rpy deg in nic_card frame: {OPPOSITE_TOOTH_NODE_OLD_RPY_DEG_NIC}")
    print("  ports:")
    for port in PORTS:
        print(f"    {port.name}:")
        print(f"      entrance: {port.entrance_path}")
        print(f"      seat: {port.seat_path}")
        print(f"      opposite tooth node: {port.opposite_tooth_node_path}")
        print(f"      opposite tooth source port: {port.opposite_tooth_source_port}")
    print(f"  port long half: {PORT_LONG_HALF:.6f} m")
    print(f"  port long total: {PORT_LONG_TOTAL:.6f} m")
    print(f"  port y half fallback: {PORT_Y_HALF_FALLBACK:.6f} m")
    print(f"  plug insertion depth fallback: {PLUG_INSERTION_DEPTH_FALLBACK:.6f} m")
    print("  local axes:")
    print(f"    long axis: {LONG_AXIS_LOCAL}")
    print(f"    tooth direction: {TOOTH_DIRECTION_LOCAL}")
    print(f"    opposite tooth direction: {OPPOSITE_TOOTH_DIRECTION_LOCAL}")
    print("    insertion axis: derived from entrance frame -> seat frame")
    print("  entrance-centered fallback keypoints:")
    for name, point in PORT_KEYPOINTS_LOCAL_FALLBACK.items():
        print(f"    {name}: {point}")
    print("  robot plug frames:")
    for note in ROBOT_PLUG_FRAME_NOTES:
        print(f"    {note}")


if __name__ == "__main__":
    print_geometry()
