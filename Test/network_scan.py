#!/usr/bin/env python3
"""
Network discovery script to find devices on the local network
"""

import socket
import subprocess
import ipaddress
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_local_ip_info():
    """Get local IP address and network information"""
    print("=== Local Network Information ===")
    
    try:
        # Get local IP by connecting to a remote address
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        
        print(f"Local IP Address: {local_ip}")
        
        # Determine network range
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
        print(f"Network Range: {network}")
        
        return local_ip, network
        
    except Exception as e:
        print(f"Error getting network info: {e}")
        return None, None

def ping_host(ip):
    """Ping a single host"""
    try:
        # Use Windows ping command
        result = subprocess.run(
            ["ping", "-n", "1", "-w", "1000", str(ip)], 
            capture_output=True, 
            text=True, 
            timeout=2
        )
        return ip, result.returncode == 0
    except:
        return ip, False

def scan_network(network, max_workers=50):
    """Scan network for active hosts"""
    print(f"\n=== Scanning Network {network} ===")
    print("This may take a moment...")
    
    active_hosts = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit ping tasks for all IPs in the network
        futures = {executor.submit(ping_host, ip): ip for ip in network.hosts()}
        
        for future in as_completed(futures):
            ip, is_active = future.result()
            if is_active:
                active_hosts.append(str(ip))
                print(f"✅ Found active host: {ip}")
    
    return active_hosts

def get_hostname(ip):
    """Try to get hostname for an IP"""
    try:
        hostname = socket.gethostbyaddr(str(ip))[0]
        return hostname
    except:
        return "Unknown"

def check_specific_ports(ip, ports=[22, 80, 443, 12355, 5000, 8000, 8080]):
    """Check specific ports on a host"""
    open_ports = []
    
    for port in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((str(ip), port))
            if result == 0:
                open_ports.append(port)
            sock.close()
        except:
            pass
    
    return open_ports

def analyze_hosts(active_hosts):
    """Analyze active hosts for more details"""
    print(f"\n=== Analyzing {len(active_hosts)} Active Hosts ===")
    
    for ip in active_hosts:
        hostname = get_hostname(ip)
        open_ports = check_specific_ports(ip)
        
        print(f"\n📍 {ip}")
        print(f"   Hostname: {hostname}")
        if open_ports:
            print(f"   Open ports: {open_ports}")
            if 12355 in open_ports:
                print(f"   🎯 PORT 12355 IS OPEN! This might be your Mac!")
        else:
            print(f"   Open ports: None detected")

def check_mac_specifically():
    """Check specifically for the Mac at 192.168.29.52"""
    print(f"\n=== Checking Mac at 192.168.29.52 ===")
    
    # Check if IP is reachable
    _, is_reachable = ping_host("192.168.29.52")
    print(f"Ping to 192.168.29.52: {'✅ Success' if is_reachable else '❌ Failed'}")
    
    if is_reachable:
        # Check hostname
        hostname = get_hostname("192.168.29.52")
        print(f"Hostname: {hostname}")
        
        # Check ports
        open_ports = check_specific_ports("192.168.29.52")
        print(f"Open ports: {open_ports}")
        
        if 12355 in open_ports:
            print("🎉 Port 12355 is open on the Mac!")
        else:
            print("❌ Port 12355 is not accessible")
    
    return is_reachable

def main():
    print("🔍 Network Device Discovery")
    print("=" * 50)
    
    # Get local network info
    local_ip, network = get_local_ip_info()
    if not local_ip or not network:
        print("❌ Could not determine network information")
        return
    
    # Check Mac specifically first
    mac_reachable = check_mac_specifically()
    
    # Scan the local network
    active_hosts = scan_network(network)
    
    if active_hosts:
        analyze_hosts(active_hosts)
        
        # Check if Mac IP is in the discovered hosts
        if "192.168.29.52" in active_hosts:
            print(f"\n✅ Mac (192.168.29.52) found in network scan!")
        else:
            print(f"\n⚠️  Mac (192.168.29.52) not found in network scan")
            if mac_reachable:
                print("   But it is reachable via ping - might be port issue")
    else:
        print("\n❌ No active hosts found on the network")
    
    print(f"\n=== Summary ===")
    print(f"Local IP: {local_ip}")
    print(f"Network: {network}")
    print(f"Active hosts found: {len(active_hosts)}")
    print(f"Mac at 192.168.29.52: {'✅ Reachable' if mac_reachable else '❌ Not reachable'}")

if __name__ == "__main__":
    main()
