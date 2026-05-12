#!/usr/bin/env python3
"""Network connection information display for Sikap-Salita live demo."""

import socket
import sys
import platform

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None

def print_network_info(port=8000):
    """Print connection information for the Sikap-Salita live demo."""
    local_ip = get_local_ip()
    hostname = socket.gethostname()
    os_name = platform.system()

    print("\n" + "="*70)
    print("  Sikap-Salita - Filipino Sign Language Recognition")
    print("  " + "-"*66)
    print("="*70)

    print("\nCONNECTION INFORMATION:")
    print(f"   Hostname: {hostname}")
    print(f"   Operating System: {os_name}")
    if local_ip:
        print(f"   Local IP: {local_ip}")

    print("\nACCESS URLs:")
    print(f"   LOCAL:   http://localhost:{port}")
    if local_ip:
        print(f"   NETWORK: http://{local_ip}:{port}")
        print(f"      (Use this URL to access from other devices on your network)")
    else:
        print(f"   NETWORK: Check your router or run `python show_network_info.py` for the IP address")

    print("\nADVANCED OPTIONS:")
    print(f"   Cloudflare Tunnel: cloudflared tunnel --url http://localhost:{port}")
    print(f"   Custom Port: uvicorn live_demo.app:app --port <PORT>")

    print("\nUSAGE:")
    print(f"   Start the app with: python run_app.py")
    print(f"   Then access using any of the URLs above")

    print("\nMODELS:")
    print("   Siformer (Feature-Isolated Transformer)")
    print("   Bi-LSTM (Bidirectional LSTM Baseline)")

    print("="*70 + "\n")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print_network_info(port)
