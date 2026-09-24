"""
Week 3: DNS Packet Capture & PCAP Parsing
==========================================
Module and CLI tool for sniffing live DNS network traffic using Scapy or tcpdump,
and extracting raw DNS queries and query metadata from saved PCAP files.
"""

import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from scapy.all import sniff, rdpcap, wrpcap, DNS, DNSQR, DNSRR, IP, UDP, TCP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


def parse_dns_packet(packet) -> Optional[Dict[str, Any]]:
    """Parse a single Scapy packet and extract DNS metadata if present."""
    if not packet.haslayer(DNS):
        return None

    dns_layer = packet.getlayer(DNS)
    record = {
        "timestamp": float(packet.time),
        "src_ip": packet[IP].src if packet.haslayer(IP) else None,
        "dst_ip": packet[IP].dst if packet.haslayer(IP) else None,
        "qr": dns_layer.qr,  # 0: query, 1: response
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


def label_records(records: List[Dict[str, Any]], label: int) -> List[Dict[str, Any]]:
    """Attach a binary traffic label: 0 for benign, 1 for malicious."""
    if label not in (0, 1):
        raise ValueError("label must be 0 (benign) or 1 (malicious)")
    return [{**record, "label": label} for record in records]


def generate_benign_records(domains: List[str], qtype: int = 1) -> List[Dict[str, Any]]:
    """Create labeled records for benign domains from a Tranco-style list.

    These records can be combined with captured malicious queries to create a
    training CSV without requiring live lookups or external network access.
    """
    timestamp = time.time()
    records = []
    for domain in domains:
        qname = domain.strip().rstrip(".")
        if qname:
            records.append({
                "timestamp": timestamp,
                "src_ip": None,
                "dst_ip": None,
                "qr": 0,
                "opcode": 0,
                "rcode": 0,
                "qname": qname,
                "qtype": qtype,
                "label": 0,
            })
    return records


def save_labeled_csv(records: List[Dict[str, Any]], output_path: str) -> Path:
    """Write DNS records and labels to a CSV file."""
    if not records:
        raise ValueError("Cannot write an empty traffic dataset")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "timestamp", "src_ip", "dst_ip", "qr", "opcode", "rcode",
        "qname", "qtype", "label",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"[+] Saved {len(records)} labeled DNS records to {output_file}")
    return output_file


def extract_queries_from_pcap(pcap_path: str) -> List[Dict[str, Any]]:
    """Read a PCAP file and extract all DNS queries into a list of dictionaries."""
    if not SCAPY_AVAILABLE:
        raise RuntimeError("Scapy is required to parse PCAP files. Install scapy first.")

    pcap_file = Path(pcap_path)
    if not pcap_file.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    print(f"[*] Reading packets from {pcap_file}...")
    packets = rdpcap(str(pcap_file))
    print(f"[+] Loaded {len(packets)} packets. Parsing DNS layers...")

    records = []
    for pkt in packets:
        parsed = parse_dns_packet(pkt)
        if parsed and parsed.get("qname"):
            records.append(parsed)

    print(f"[+] Extracted {len(records)} DNS queries from PCAP.")
    return records


def live_capture(
    interface: str = "lo",
    bpf_filter: str = "udp port 53",
    count: int = 100,
    timeout: int = 60,
    output_pcap: Optional[str] = None
):
    """Capture live DNS traffic on the specified interface."""
    if not SCAPY_AVAILABLE:
        raise RuntimeError("Scapy is required for live network capture.")

    print(f"[*] Sniffing on {interface} with filter '{bpf_filter}' (count={count}, timeout={timeout}s)...")
    captured_packets = sniff(
        iface=interface,
        filter=bpf_filter,
        count=count,
        timeout=timeout
    )

    print(f"[+] Sniffed {len(captured_packets)} packets.")
    if output_pcap and len(captured_packets) > 0:
        out_path = Path(output_pcap)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wrpcap(str(out_path), captured_packets)
        print(f"[+] Captured packets saved to {out_path}")

    return captured_packets


def main():
    parser = argparse.ArgumentParser(description="Capture or parse DNS network traffic.")
    parser.add_argument("--pcap", help="Path to PCAP file to parse")
    parser.add_argument("--interface", default="lo", help="Network interface to sniff on")
    parser.add_argument("--filter", default="udp port 53", help="BPF capture filter")
    parser.add_argument("--count", type=int, default=100, help="Number of packets to capture")
    parser.add_argument("--timeout", type=int, default=30, help="Capture timeout in seconds")
    parser.add_argument("--out-pcap", help="Output path for captured PCAP file")
    parser.add_argument("--output-csv", default="data/processed/labeled_traffic.csv", help="Labeled CSV output path")
    parser.add_argument("--label", type=int, choices=[0, 1], default=1, help="Label captured records: 0 benign, 1 malicious")
    parser.add_argument("--benign-file", help="Text or Tranco CSV file of benign domains to add with label 0")
    parser.add_argument("--benign-limit", type=int, default=100, help="Maximum benign domains to add")

    args = parser.parse_args()

    records = []
    if args.pcap:
        records = extract_queries_from_pcap(args.pcap)
        records = label_records(records, args.label)
        for r in records[:10]:
            print(f"    [{r['timestamp']:.2f}] {r['src_ip']} -> {r['dst_ip']} | Q: {r['qname']} (Type {r['qtype']})")
        if len(records) > 10:
            print(f"    ... and {len(records) - 10} more queries.")
    else:
        captured_packets = live_capture(
            interface=args.interface,
            bpf_filter=args.filter,
            count=args.count,
            timeout=args.timeout,
            output_pcap=args.out_pcap
        )
        records = label_records(
            [record for packet in captured_packets if (record := parse_dns_packet(packet))],
            args.label,
        )

    if args.benign_file:
        benign_path = Path(args.benign_file)
        if not benign_path.exists():
            raise FileNotFoundError(f"Benign domain file not found: {benign_path}")
        domains = []
        for line in benign_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.lower().startswith(("rank,", "domain,")):
                domains.append(value.split(",", 1)[-1])
            if len(domains) >= args.benign_limit:
                break
        records.extend(generate_benign_records(domains))

    if records:
        save_labeled_csv(records, args.output_csv)


if __name__ == "__main__":
    main()
