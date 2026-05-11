"""
mitigation_engine.py
─────────────────────
Handles automatic DDoS mitigation by installing OpenFlow rules
into SDN switches via the Ryu controller.

Mitigation actions:
  1. DROP       — block all traffic from attacker IP
  2. RATE LIMIT — throttle traffic to max_rate kbps
  3. REROUTE    — redirect to honeypot / alternate path

Usage (called from ddos_controller.py):
    from mitigation.mitigation_engine import MitigationEngine
    engine = MitigationEngine()
    engine.block_ip(datapath, src_ip='10.0.0.99')
"""

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('MitigationEngine')

# Trust table: tracks blocked IPs and their block time
BLOCKED_IPS = {}
BLOCK_DURATION_SEC = 60   # auto-unblock after 60 seconds


class MitigationEngine:

    def __init__(self):
        self.blocked = BLOCKED_IPS

    # ── DROP Rule ────────────────────────────────────────────
    def block_ip(self, datapath, src_ip, priority=200, duration=60):
        """
        Installs a DROP flow rule for the given source IP.

        Parameters:
            datapath  : Ryu datapath object (the switch)
            src_ip    : attacker IP address (string)
            priority  : OpenFlow rule priority (higher = matched first)
            duration  : how long to block in seconds
        """
        if src_ip in self.blocked:
            # Check if block has expired
            if time.time() - self.blocked[src_ip] < BLOCK_DURATION_SEC:
                logger.debug(f"{src_ip} already blocked.")
                return
            else:
                del self.blocked[src_ip]

        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser

        # Match on source IP (IPv4)
        match = parser.OFPMatch(
            eth_type=0x0800,     # IPv4
            ipv4_src=src_ip
        )

        # Empty actions list = DROP
        actions = []
        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions
        )]

        # Install the flow rule
        mod = parser.OFPFlowMod(
            datapath   = datapath,
            priority   = priority,
            match      = match,
            instructions = inst,
            hard_timeout = duration,     # auto-expire after `duration` sec
            idle_timeout = 0
        )
        datapath.send_msg(mod)

        self.blocked[src_ip] = time.time()
        logger.warning(
            f"[BLOCK] {src_ip} → DROP rule installed "
            f"(expires in {duration}s)"
        )

    # ── RATE LIMIT Rule ──────────────────────────────────────
    def rate_limit_ip(self, datapath, src_ip, max_rate_kbps=100):
        """
        Limits bandwidth from an IP using an OpenFlow meter.
        Note: Requires OVS with meter support.

        Parameters:
            datapath     : Ryu datapath object
            src_ip       : IP to rate-limit
            max_rate_kbps: maximum allowed bandwidth in kbps
        """
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser

        # Add a meter (rate-limit = max_rate_kbps kbps)
        bands = [parser.OFPMeterBandDrop(
            rate=max_rate_kbps,
            burst_size=10
        )]
        meter_mod = parser.OFPMeterMod(
            datapath  = datapath,
            command   = ofproto.OFPMC_ADD,
            flags     = ofproto.OFPMF_KBPS,
            meter_id  = 1,
            bands     = bands
        )
        datapath.send_msg(meter_mod)

        # Match source IP and apply meter
        match   = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
        inst    = [
            parser.OFPInstructionMeter(meter_id=1),
            parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, [])
        ]
        mod = parser.OFPFlowMod(
            datapath     = datapath,
            priority     = 150,
            match        = match,
            instructions = inst,
            hard_timeout = 120
        )
        datapath.send_msg(mod)
        logger.warning(
            f"[RATE LIMIT] {src_ip} limited to {max_rate_kbps} kbps"
        )

    # ── Unblock ───────────────────────────────────────────────
    def unblock_ip(self, datapath, src_ip):
        """Remove block rule for an IP (manual unblock)."""
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser

        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
        mod   = parser.OFPFlowMod(
            datapath  = datapath,
            command   = ofproto.OFPFC_DELETE,
            out_port  = ofproto.OFPP_ANY,
            out_group = ofproto.OFPG_ANY,
            match     = match
        )
        datapath.send_msg(mod)
        self.blocked.pop(src_ip, None)
        logger.info(f"[UNBLOCK] {src_ip} rule removed.")

    def get_blocked_ips(self):
        """Return list of currently blocked IPs."""
        now = time.time()
        active = {
            ip: t for ip, t in self.blocked.items()
            if now - t < BLOCK_DURATION_SEC
        }
        return list(active.keys())
