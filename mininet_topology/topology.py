"""
topology.py
───────────
Custom Mininet topology for SDN DDoS detection testing.

Network layout:
                        [Server h3]
                             |
   [h1] ── [s1] ── [s2] ── [s3]
   [h2] ──/          |
[attacker] ──────────┘

Hosts:
  h1         10.0.0.1  — legitimate user
  h2         10.0.0.2  — legitimate user
  attacker   10.0.0.99 — simulated attacker
  server     10.0.0.10 — target server (victim)

Controller: Ryu running on 127.0.0.1:6653

Run with:
    sudo python3 topology.py

(Start the Ryu controller FIRST in another terminal)
"""

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import time


def create_topology():
    setLogLevel('info')

    info("*** Creating SDN network\n")
    net = Mininet(
        switch     = OVSSwitch,
        controller = RemoteController,
        link       = TCLink,          # enables bandwidth/delay settings
        autoSetMacs = True
    )

    # ── Controller ────────────────────────────────────────────
    info("*** Adding Ryu controller\n")
    c0 = net.addController(
        'c0',
        controller = RemoteController,
        ip         = '127.0.0.1',
        port       = 6653
    )

    # ── Switches ──────────────────────────────────────────────
    info("*** Adding switches\n")
    s1 = net.addSwitch('s1', protocols='OpenFlow13')
    s2 = net.addSwitch('s2', protocols='OpenFlow13')
    s3 = net.addSwitch('s3', protocols='OpenFlow13')

    # ── Hosts ─────────────────────────────────────────────────
    info("*** Adding hosts\n")
    h1       = net.addHost('h1',       ip='10.0.0.1/24')
    h2       = net.addHost('h2',       ip='10.0.0.2/24')
    attacker = net.addHost('attacker', ip='10.0.0.99/24')
    server   = net.addHost('server',   ip='10.0.0.10/24')

    # ── Links ─────────────────────────────────────────────────
    info("*** Creating links\n")
    net.addLink(h1,       s1, bw=100, delay='1ms')
    net.addLink(h2,       s1, bw=100, delay='1ms')
    net.addLink(attacker, s1, bw=100, delay='1ms')   # attacker on same switch
    net.addLink(s1,       s2, bw=1000, delay='1ms')
    net.addLink(s2,       s3, bw=1000, delay='1ms')
    net.addLink(server,   s3, bw=100,  delay='1ms')

    # ── Start Network ─────────────────────────────────────────
    info("*** Starting network\n")
    net.build()
    c0.start()
    s1.start([c0])
    s2.start([c0])
    s3.start([c0])

    info("*** Waiting for controller to connect...\n")
    time.sleep(3)

    # ── Test connectivity ─────────────────────────────────────
    info("*** Testing basic connectivity (h1 → server)\n")
    net.pingAll()

    # ── Instructions ──────────────────────────────────────────
    info("\n")
    info("=" * 55 + "\n")
    info(" Network is READY. Ryu controller is monitoring.\n")
    info("=" * 55 + "\n")
    info(" To simulate a DDoS attack, run in this CLI:\n")
    info("   mininet> attacker hping3 -S --flood -V -p 80 10.0.0.10\n")
    info("\n")
    info(" To generate normal traffic:\n")
    info("   mininet> h1 ping -c 10 10.0.0.10\n")
    info("   mininet> h2 iperf -s &  (on server)\n")
    info("   mininet> h1 iperf -c 10.0.0.10\n")
    info("=" * 55 + "\n")

    # ── CLI ───────────────────────────────────────────────────
    CLI(net)

    # ── Cleanup ───────────────────────────────────────────────
    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    create_topology()
