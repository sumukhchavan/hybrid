"""
ddos_controller.py
──────────────────
Main Ryu SDN Controller application.

What it does:
  1. Connects to all OpenFlow switches in the Mininet topology.
  2. Polls each switch for flow statistics every 5 seconds.
  3. Extracts features from flow stats (packet count, byte count, etc.)
  4. Feeds features into HybridDetector (SVM + RF ensemble).
  5. If DDoS detected → calls MitigationEngine to install DROP rules.
  6. Logs all attacks via AlertSystem.

Run with:
    ryu-manager ryu_controller/ddos_controller.py

(Make sure Mininet topology is also running in another terminal)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import logging
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ipv4

from detection.hybrid_detector import HybridDetector
from mitigation.mitigation_engine import MitigationEngine
from alerts.alert_system import AlertSystem

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(name)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('DDoSController')

POLL_INTERVAL   = 5    # seconds between flow stat requests
ATTACK_LABEL    = 'DDoS'


class DDoSController(app_manager.RyuApp):
    """
    Ryu OpenFlow 1.3 Controller with integrated Hybrid SVM-RF DDoS detection.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DDoSController, self).__init__(*args, **kwargs)
        self.datapaths   = {}       # dpid → datapath
        self.mac_to_port = {}       # for basic L2 switching

        # Load ML model
        try:
            self.detector = HybridDetector()
            logger.info("Hybrid SVM-RF model loaded successfully.")
        except FileNotFoundError:
            logger.error(
                "Models not found! Train the model first:\n"
                "  cd ml_model && python3 train_model.py"
            )
            self.detector = None

        self.mitigator = MitigationEngine()
        self.alerter   = AlertSystem()

        # Start background polling thread
        self.monitor_thread = hub.spawn(self._monitor)

    # ── Switch Handshake ─────────────────────────────────────
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser

        # Install table-miss flow: send all unmatched packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER
        )]
        self._add_flow(datapath, priority=0, match=match, actions=actions)
        self.datapaths[datapath.id] = datapath
        logger.info(f"Switch connected: dpid={datapath.id}")

    # ── Packet-In: Basic L2 Switching ────────────────────────
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg      = ev.msg
        datapath = msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        in_port  = msg.match['in_port']

        pkt     = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        if eth_pkt is None:
            return

        dst  = eth_pkt.dst
        src  = eth_pkt.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = (
            self.mac_to_port[dpid][dst]
            if dst in self.mac_to_port[dpid]
            else ofproto.OFPP_FLOOD
        )

        actions = [parser.OFPActionOutput(out_port)]
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self._add_flow(datapath, priority=1, match=match, actions=actions)

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        )
        datapath.send_msg(out)

    # ── Background Monitor ────────────────────────────────────
    def _monitor(self):
        """Polls flow statistics from all switches every POLL_INTERVAL seconds."""
        while True:
            for dp in list(self.datapaths.values()):
                self._request_flow_stats(dp)
            hub.sleep(POLL_INTERVAL)

    def _request_flow_stats(self, datapath):
        parser  = datapath.ofproto_parser
        req     = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # ── Flow Stats Reply: Detection ───────────────────────────
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        if self.detector is None:
            return

        datapath = ev.msg.datapath
        flows    = []
        meta     = []   # store src IP for mitigation

        for stat in ev.msg.body:
            # Extract features from each flow
            features = self._extract_features(stat)
            if features is not None:
                flows.append(features)
                # Try to get source IP from match fields
                src_ip = stat.match.get('ipv4_src', None)
                meta.append(src_ip)

        if not flows:
            return

        # Batch predict
        results = self.detector.predict_batch(np.array(flows))

        for i, result in enumerate(results):
            if result['is_attack']:
                src_ip = meta[i] if i < len(meta) else 'Unknown'
                conf   = result['confidence']

                logger.warning(
                    f"[ATTACK] DDoS detected! "
                    f"src={src_ip}  confidence={conf:.2%}"
                )

                # Mitigate
                if src_ip and src_ip != 'Unknown':
                    self.mitigator.block_ip(datapath, src_ip)

                # Alert
                self.alerter.log_attack(
                    src_ip=src_ip or 'Unknown',
                    attack_type='DDoS',
                    confidence=conf
                )
                # Uncomment below to also send email:
                # self.alerter.send_alert(src_ip=src_ip, confidence=conf)

    # ── Feature Extraction ────────────────────────────────────
    def _extract_features(self, stat):
        """
        Extracts numerical features from an OFPFlowStats object.

        Returns a list of floats, or None if the stat is a table-miss entry.
        """
        try:
            # Skip table-miss entries (priority 0, no real traffic features)
            if stat.priority == 0:
                return None

            duration    = stat.duration_sec + stat.duration_nsec * 1e-9
            pkt_count   = stat.packet_count
            byte_count  = stat.byte_count
            pkt_per_sec = pkt_count / duration if duration > 0 else 0
            byt_per_sec = byte_count / duration if duration > 0 else 0

            # Basic feature vector
            # (Extend this list to match your dataset's features)
            features = [
                pkt_count,
                byte_count,
                stat.duration_sec,
                stat.duration_nsec,
                pkt_per_sec,
                byt_per_sec,
                stat.priority,
                stat.idle_timeout,
                stat.hard_timeout,
            ]
            return features

        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None

    # ── Helper: Add Flow Rule ─────────────────────────────────
    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        inst    = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions
        )]
        mod = parser.OFPFlowMod(
            datapath     = datapath,
            priority     = priority,
            match        = match,
            instructions = inst,
            idle_timeout = idle_timeout,
            hard_timeout = hard_timeout
        )
        datapath.send_msg(mod)
