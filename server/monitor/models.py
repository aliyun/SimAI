"""Data models for network monitoring dashboard."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class NodeType(str, Enum):
    OXC = "OXC"
    SPINE = "SPINE"
    LEAF = "LEAF"
    SERVER = "SERVER"
    POD = "POD"


class NodeStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ERROR = "error"
    OFFLINE = "offline"


class LinkStatus(str, Enum):
    UP = "up"
    DEGRADED = "degraded"
    DOWN = "down"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    LINK_UTILIZATION_HIGH = "link_utilization_high"
    NODE_OFFLINE = "node_offline"
    LINK_DOWN = "link_down"
    TRAFFIC_ANOMALY = "traffic_anomaly"
    PACKET_LOSS = "packet_loss"


@dataclass(frozen=True)
class PortMapping:
    side_a_ports: tuple
    side_b_ports: tuple


@dataclass(frozen=True)
class TrafficMetrics:
    utilization_percent: float
    bandwidth_gbps: float
    max_bandwidth_gbps: float
    multiplier: str
    secondary_percent: float
    packets_per_second: int
    error_rate: float
    timestamp: float


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str
    node_ip: Optional[str]
    status: str
    pod_id: Optional[str]
    server_type: Optional[str] = None
    super_node_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Link:
    link_id: str
    source_node_id: str
    target_node_id: str
    status: str
    port_mapping: Optional[dict] = None
    traffic: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Alert:
    alert_id: str
    alert_type: str
    severity: str
    title: str
    message: str
    created_at: float
    source_node_id: Optional[str] = None
    source_link_id: Optional[str] = None
    pod_id: Optional[str] = None
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
