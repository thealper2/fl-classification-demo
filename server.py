import numpy as np
import pandas as pd
import os
import sys
import flwr
import argparse
import ipaddress

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Classification Demo')
    parser.add_argument("--address", help="IP address", default="0.0.0.0", type=str)
    parser.add_argument("--port", help="Serving port", default=8000, type=int)
    parser.add_argument("--rounds", help="Number of rounds", default=10, type=int)
    args = parser.parse_args()

    try:
        ipaddress.ip_address(args.address)
    except ValueError:
        sys.exit(f"Wrong IP address: {args.address}")

    if args.port < 0 or args.port > 65535:
        sys.exit(f"Wrong port number: {args.port}")

    if args.rounds < 0:
        sys.exit(f"Wrong round number: {args.rounds}")

    flwr.server.start_server(
        server_address=f"{args.address}:{args.port}",
        config=flwr.server.ServerConfig(num_rounds=args.rounds),
    )