"""
Week 3: DNS Traffic Capture & Synthesis Engine
==============================================
Generates and captures realistic DNS network traffic combining:
1. Benign web lookups (sampled from Tranco top-1M domain list).
2. Burst DNS exfiltration (rapid high-frequency encoded query spikes).
3. Low-and-slow DNS exfiltration (evasive covert trickle with variable jitter).
4. Packet capture via tcpdump or Scapy with export to labeled PCAP and CSV.
"""

import os
import sys
import time
import random
import argparse
import subprocess
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

try:
    from scapy.all import IP, UDP, DNS, DNSQR, DNSRR, sniff, rdpcap, wrpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

try:
    from src.dns_simulator import DNSExfiltrationSimulator, validate_rfc_compliance
except ImportError:
    from dns_simulator import DNSExfiltrationSimulator, validate_rfc_compliance


def load_benign_domains(
    tranco_path: str = "data/raw/top-1m.csv",
    limit: int = 500
) -> List[str]:
    """Load sample benign domains from Tranco list or fallback to high-profile web services."""
    path = Path(tranco_path)
    if path.exists() and path.stat().st_size > 0:
        domains = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                dom = parts[-1].strip().lower()
                if dom and not dom.startswith(("rank", "domain")):
                    domains.append(dom)
                if len(domains) >= limit:
                    break
        if domains:
            return domains

    # Fallback curated list of major benign domains
    fallback = [
        "google.com", "cloudflare.com", "apple.com", "microsoft.com",
        "github.com", "wikipedia.org", "amazon.com", "netflix.com",
        "cnn.com", "reddit.com", "linkedin.com", "yahoo.com",
        "bing.com", "spotify.com", "dropbox.com", "slack.com",
        "zoom.us", "mozilla.org", "adobe.com", "salesforce.com"
    ]
    return fallback * (limit // len(fallback) + 1)


def generate_synthesized_traffic(
    benign_count: int = 200,
    burst_count: int = 150,
    low_slow_count: int = 50,
    tranco_file: str = "data/raw/top-1m.csv",
    target_tunnel: str = "tunnel.corp-exfil.net",
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Synthesize a realistic mixed DNS traffic session comprising:
    - Benign web queries across corporate / client devices.
    - Burst exfiltration: high query rate, Base32/Hex encoded secrets.
    - Low-and-slow exfiltration: spaced intervals, Base32 covert signaling.
    
    Returns:
        (records: List[Dict], scapy_packets: List[Packet])
    """
    random.seed(seed)
    records: List[Dict[str, Any]] = []
    packets: List[Any] = []

    benign_domains = load_benign_domains(tranco_file, limit=max(benign_count * 2, 500))

    # Base clock
    current_time = time.time() - 3600.0  # 1 hour ago
    sim_b32 = DNSExfiltrationSimulator(target_domain=target_tunnel, encoding="base32")
    sim_hex = DNSExfiltrationSimulator(target_domain=target_tunnel, encoding="hex")

    client_ips = [f"192.168.1.{i}" for i in range(10, 80)]
    infected_burst_ip = "192.168.1.105"
    infected_slow_ip = "192.168.1.188"
    resolver_ip = "1.1.1.1"

    # Pre-generate exfil queries ensuring exact counts requested
    burst_queries = []
    burst_bytes_multiplier = 35
    while len(burst_queries) < burst_count:
        burst_queries = sim_hex.generate_queries(os.urandom(max(burst_count, 1) * burst_bytes_multiplier))
        burst_bytes_multiplier += 20
    burst_queries = burst_queries[:burst_count]

    slow_queries = []
    slow_bytes_multiplier = 35
    while len(slow_queries) < low_slow_count:
        slow_queries = sim_b32.generate_queries(os.urandom(max(low_slow_count, 1) * slow_bytes_multiplier))
        slow_bytes_multiplier += 20
    slow_queries = slow_queries[:low_slow_count]

    qtype_num_map = {"A": 1, "AAAA": 28, "TXT": 16, "CNAME": 5}

    print(f"[*] Synthesizing mixed DNS traffic:")
    print(f"    - Benign lookups:      {benign_count}")
    print(f"    - Burst exfiltration:  {len(burst_queries)} queries (IP: {infected_burst_ip})")
    print(f"    - Low-and-slow exfil:  {len(slow_queries)} queries (IP: {infected_slow_ip})")

    # 1. Timeline interleaving
    events = []

    # Benign events spread over session
    b_time = current_time
    for i in range(benign_count):
        b_time += random.uniform(0.5, 4.0)
        domain = random.choice(benign_domains)
        if random.random() < 0.25:
            domain = f"cdn-{random.randint(1,9)}.{domain}"
        qtype = random.choice(["A", "A", "A", "AAAA", "CNAME"])
        events.append((b_time, "benign", random.choice(client_ips), domain, qtype, 0))

    # Low-and-slow events: periodic trickles (every 25 - 60s)
    slow_time = current_time + random.uniform(10.0, 30.0)
    for q in slow_queries:
        slow_time += random.uniform(20.0, 45.0)
        events.append((slow_time, "low_and_slow_exfil", infected_slow_ip, q, "TXT", 1))

    # Burst events: concentrated spike midway through session
    burst_start = current_time + 400.0
    burst_time = burst_start
    for q in burst_queries:
        burst_time += random.uniform(0.01, 0.05)  # 20 - 100 queries/sec
        events.append((burst_time, "burst_exfil", infected_burst_ip, q, "TXT", 1))

    # Sort all events chronologically
    events.sort(key=lambda x: x[0])

    # 2. Build Scapy packets and records
    for ts, attack_type, src_ip, qname, qtype_str, label in events:
        qtype_num = qtype_num_map.get(qtype_str, 1)
        record = {
            "timestamp": round(ts, 4),
            "src_ip": src_ip,
            "dst_ip": resolver_ip,
            "qr": 0,
            "opcode": 0,
            "rcode": 0,
            "qname": qname,
            "qtype": qtype_num,
            "attack_type": attack_type,
            "label": label,
        }
        records.append(record)

        if SCAPY_AVAILABLE:
            pkt = (
                IP(src=src_ip, dst=resolver_ip)
                / UDP(sport=random.randint(2000, 65000), dport=53)
                / DNS(rd=1, qd=DNSQR(qname=qname, qtype=qtype_num))
            )
            pkt.time = ts
            packets.append(pkt)

    print(f"[✓] Synthesis completed: {len(records)} total queries synthesized.")
    return records, packets


def parse_dns_packet(packet) -> Optional[Dict[str, Any]]:
    """Parse a single Scapy packet and extract DNS query metadata."""
    if not packet.haslayer(DNS):
        return None

    dns_layer = packet.getlayer(DNS)
    record = {
        "timestamp": float(packet.time),
        "src_ip": packet[IP].src if packet.haslayer(IP) else None,
        "dst_ip": packet[IP].dst if packet.haslayer(IP) else None,
        "qr": dns_layer.qr,
        "opcode": dns_layer.opcode,
        "rcode": dns_layer.rcode,
        "qname": None,
        "qtype": None,
    }

    if packet.haslayer(DNSQR) and dns_layer.qd:
        try:
            record["qname"] = dns_layer.qd.qname.decode("utf-8", errors="ignore").rstrip(".")
            record["qtype"] = dns_layer.qd.qtype
        except Exception:
            pass

    return record


def extract_queries_from_pcap(pcap_path: str) -> List[Dict[str, Any]]:
    """Extract all DNS queries from an existing PCAP capture file."""
    if not SCAPY_AVAILABLE:
        raise RuntimeError("Scapy is required to parse PCAP files.")

    pcap_file = Path(pcap_path)
    if not pcap_file.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    print(f"[*] Reading PCAP file: {pcap_file} ({pcap_file.stat().st_size / 1024:.1f} KB)...")
    packets = rdpcap(str(pcap_file))
    records = []

    for pkt in packets:
        parsed = parse_dns_packet(pkt)
        if parsed and parsed.get("qname"):
            records.append(parsed)

    print(f"[✓] Extracted {len(records)} DNS queries from {pcap_file.name}")
    return records


def label_dns_records(
    records: List[Dict[str, Any]],
    tunnel_domain: str = "tunnel.",
    default_label: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Assign ground-truth labels (0: benign, 1: exfil) to DNS records.
    Uses tunnel domain signatures or default label override.
    """
    labeled = []
    for r in records:
        rec = dict(r)
        if default_label is not None:
            rec["label"] = default_label
        elif "label" in rec:
            pass
        else:
            qname = rec.get("qname", "").lower()
            # Heuristic detection for ground truth labeling:
            # Exfiltration queries typically contain the tunnel subdomain or high-length sequence headers
            if tunnel_domain in qname or "sq00" in qname or len(qname.split(".")[0]) > 45:
                rec["label"] = 1
                rec["attack_type"] = "exfiltration"
            else:
                rec["label"] = 0
                rec["attack_type"] = "benign"
        labeled.append(rec)
    return labeled


def capture_with_tcpdump(
    interface: str = "lo",
    output_pcap: str = "data/raw/tcpdump_capture.pcap",
    count: int = 50,
    timeout: int = 15,
    bpf_filter: str = "udp port 53"
) -> bool:
    """
    Capture live packets via tcpdump CLI tool.
    Falls back gracefully with diagnostic output if permissions are missing.
    """
    pcap_path = Path(output_pcap)
    pcap_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "tcpdump",
        "-i", interface,
        "-c", str(count),
        "-w", str(pcap_path),
        "-nn",
        bpf_filter
    ]

    print(f"[*] Starting tcpdump capture: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        if proc.returncode == 0 and pcap_path.exists() and pcap_path.stat().st_size > 0:
            print(f"[✓] tcpdump captured packets successfully to {pcap_path}")
            return True
        else:
            print(f"[!] tcpdump exited with code {proc.returncode}.")
            if proc.stderr:
                print(f"    tcpdump stderr: {proc.stderr.strip()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[!] tcpdump timed out after {timeout} seconds.")
        return pcap_path.exists() and pcap_path.stat().st_size > 0
    except Exception as e:
        print(f"[!] tcpdump execution failed: {e}")
        return False


def live_capture_scapy(
    interface: str = "lo",
    output_pcap: Optional[str] = None,
    count: int = 50,
    timeout: int = 15,
    bpf_filter: str = "udp port 53"
):
    """Capture live packets using Scapy's native sniff engine."""
    if not SCAPY_AVAILABLE:
        raise RuntimeError("Scapy is required for native live capture.")

    print(f"[*] Sniffing on '{interface}' (filter: '{bpf_filter}', count={count}, timeout={timeout}s)...")
    packets = sniff(iface=interface, filter=bpf_filter, count=count, timeout=timeout)
    print(f"[✓] Scapy captured {len(packets)} packets.")

    if output_pcap and len(packets) > 0:
        out_path = Path(output_pcap)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wrpcap(str(out_path), packets)
        print(f"[✓] Saved captured packets to {out_path}")

    return packets


def save_labeled_csv(records: List[Dict[str, Any]], output_path: str) -> Path:
    """Save records to structured CSV file."""
    if not records:
        raise ValueError("Cannot write empty records list to CSV")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    # Ensure consistent column ordering
    preferred_cols = ["timestamp", "src_ip", "dst_ip", "qr", "opcode", "rcode", "qname", "qtype", "attack_type", "label"]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]
    df.to_csv(out_file, index=False)
    print(f"[✓] Saved {len(df)} labeled records to {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Week 3: Traffic Capture & Synthesis Engine (Burst, Low-and-Slow, Benign, tcpdump, PCAP)"
    )
    parser.add_argument("--synthesize", action="store_true", help="Generate mixed burst, low-and-slow, and benign traffic")
    parser.add_argument("--benign-count", type=int, default=300, help="Number of benign queries to synthesize")
    parser.add_argument("--burst-count", type=int, default=200, help="Number of burst exfiltration queries")
    parser.add_argument("--low-slow-count", type=int, default=80, help="Number of low-and-slow exfiltration queries")
    parser.add_argument("--pcap", help="Parse and label an existing PCAP capture file")
    parser.add_argument("--tcpdump", action="store_true", help="Capture live traffic using tcpdump")
    parser.add_argument("--interface", default="lo", help="Network interface for live capture")
    parser.add_argument("--count", type=int, default=100, help="Number of packets to capture")
    parser.add_argument("--timeout", type=int, default=20, help="Capture timeout in seconds")
    parser.add_argument("--pcap-out", default="data/raw/traffic_capture.pcap", help="Path for output PCAP file")
    parser.add_argument("--output-csv", default="data/processed/labeled_traffic.csv", help="Path for labeled output CSV")
    parser.add_argument("--tunnel-domain", default="tunnel.", help="Tunnel domain substring for heuristic ground-truth labeling")
    parser.add_argument("--label", type=int, choices=[0, 1], help="Force label for all records (0: benign, 1: malicious)")

    args = parser.parse_args()

    records = []
    packets = []

    if args.synthesize:
        records, packets = generate_synthesized_traffic(
            benign_count=args.benign_count,
            burst_count=args.burst_count,
            low_slow_count=args.low_slow_count,
            target_tunnel="tunnel.corp-exfil.net"
        )
        if SCAPY_AVAILABLE and packets and args.pcap_out:
            pcap_path = Path(args.pcap_out)
            pcap_path.parent.mkdir(parents=True, exist_ok=True)
            wrpcap(str(pcap_path), packets)
            print(f"[✓] Saved {len(packets)} synthesized packets to PCAP: {pcap_path}")

        save_labeled_csv(records, args.output_csv)

    elif args.tcpdump:
        success = capture_with_tcpdump(
            interface=args.interface,
            output_pcap=args.pcap_out,
            count=args.count,
            timeout=args.timeout,
        )
        if not success:
            print("[!] Falling back to Scapy sniffer for packet capture...")
            try:
                scapy_pkts = live_capture_scapy(
                    interface=args.interface,
                    output_pcap=args.pcap_out,
                    count=args.count,
                    timeout=args.timeout
                )
                if scapy_pkts:
                    records = extract_queries_from_pcap(args.pcap_out)
                    records = label_dns_records(records, tunnel_domain=args.tunnel_domain, default_label=args.label)
                    save_labeled_csv(records, args.output_csv)
            except Exception as e:
                print(f"[!] Scapy capture error: {e}")
                print("    Synthesizing traffic session instead to ensure workflow continuity...")
                records, packets = generate_synthesized_traffic(benign_count=100, burst_count=80, low_slow_count=30)
                if SCAPY_AVAILABLE and packets:
                    wrpcap(args.pcap_out, packets)
                save_labeled_csv(records, args.output_csv)
        else:
            records = extract_queries_from_pcap(args.pcap_out)
            records = label_dns_records(records, tunnel_domain=args.tunnel_domain, default_label=args.label)
            save_labeled_csv(records, args.output_csv)

    elif args.pcap:
        records = extract_queries_from_pcap(args.pcap)
        records = label_dns_records(records, tunnel_domain=args.tunnel_domain, default_label=args.label)
        save_labeled_csv(records, args.output_csv)

    else:
        # Default action: synthesize a complete dataset session
        print("[*] No specific mode chosen. Synthesizing full benchmark session...")
        records, packets = generate_synthesized_traffic(
            benign_count=args.benign_count,
            burst_count=args.burst_count,
            low_slow_count=args.low_slow_count
        )
        if SCAPY_AVAILABLE and packets and args.pcap_out:
            wrpcap(args.pcap_out, packets)
            print(f"[✓] Exported synthesized PCAP to {args.pcap_out}")
        save_labeled_csv(records, args.output_csv)


if __name__ == "__main__":
    main()

