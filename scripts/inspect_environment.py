#!/usr/bin/env python3
"""
Environment inspection script for OLORIC.
Checks system specifications, dependencies, and GPU availability.
"""
import os
import sys
import platform
import subprocess
import pkg_resources
from typing import Dict, Any
import torch

def check_python_version() -> Dict[str, Any]:
    """Check Python version."""
    version = sys.version_info
    return {
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "meets_requirement": version.major == 3 and version.minor >= 11,
        "requirement": "Python 3.11+"
    }

def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": []
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            availability["gpus"].append({
                "id": i,
                "name": gpu_props.name,
                "total_memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                "major": gpu_props.major,
                "minor": gpu_props.minor
            })
    return availability

def check_dependencies() -> Dict[str, Any]:
    """Check key dependencies."""
    required_packages = {
        "torch": "2.0.0",
        "transformers": "4.30.0",
        "peft": "0.5.0",
        "bitsandbytes": "0.41.0",
        "datasets": "2.10.0",
        "pydantic": "2.0.0",
        "yaml": "6.0"
    }
    
    results = {}
    for package, min_version in required_packages.items():
        try:
            if package == "yaml":
                # PyYAML is imported as yaml
                installed_version = pkg_resources.get_distribution("PyYAML").version
            else:
                installed_version = pkg_resources.get_distribution(package).version
            
            # Simple version check (doesn't handle complex version specs)
            results[package] = {
                "installed": installed_version,
                "required": min_version,
                "status": "installed"  # Simplified
            }
        except pkg_resources.DistributionNotFound:
            results[package] = {
                "installed": None,
                "required": min_version,
                "status": "missing"
            }
        except Exception as e:
            results[package] = {
                "installed": "unknown",
                "required": min_version,
                "status": f"error: {e}"
            }
    
    return results

def check_disk_space() -> Dict[str, Any]:
    """Check available disk space."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    return {
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
        "usage_percent": round((used / total) * 100, 2)
    }

def main():
    """Main inspection function."""
    print("=" * 60)
    print("OLORIC Environment Inspection")
    print("=" * 60)
    
    # Python version
    print("\n1. Python Version:")
    py_info = check_python_version()
    print(f"   Version: {py_info['version']}")
    print(f"   Requirement: {py_info['requirement']}")
    print(f"   Status: {'✓ PASS' if py_info['meets_requirement'] else '✗ FAIL'}")
    
    # GPU availability
    print("\n2. GPU Availability:")
    gpu_info = check_gpu_availability()
    print(f"   CUDA Available: {'✓ YES' if gpu_info['cuda_available'] else '✗ NO'}")
    print(f"   GPU Count: {gpu_info['gpu_count']}")
    for gpu in gpu_info["gpus"]:
        print(f"   GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']} GB)")
    
    # Dependencies
    print("\n3. Dependencies:")
    deps = check_dependencies()
    for package, info in deps.items():
        status_symbol = "✓" if info["status"] == "installed" else "✗"
        print(f"   {status_symbol} {package}: {info['installed']} (required: {info['required']})")
    
    # Disk space
    print("\n4. Disk Space:")
    disk_info = check_disk_space()
    print(f"   Total: {disk_info['total_gb']} GB")
    print(f"   Used: {disk_info['used_gb']} GB ({disk_info['usage_percent']}%)")
    print(f"   Free: {disk_info['free_gb']} GB")
    
    # System info
    print("\n5. System Information:")
    print(f"   OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)
    
    ready = True
    ready &= py_info["meets_requirement"]
    ready &= gpu_info["cuda_available"] and gpu_info["gpu_count"] > 0
    ready &= all(info["status"] == "installed" for info in deps.values())
    ready &= disk_info["free_gb"] > 10  # At least 10 GB free
    
    print(f"System Ready for OLORIC: {'✓ YES' if ready else '✗ NO'}")
    if not ready:
        print("\nIssues to address:")
        if not py_info["meets_requirement"]:
            print("  - Python version does not meet requirement (3.11+)")
        if not gpu_info["cuda_available"]:
            print("  - No CUDA-capable GPU detected")
        if gpu_info["gpu_count"] == 0:
            print("  - No GPUs available")
        missing_deps = [pkg for pkg, info in deps.items() if info["status"] != "installed"]
        if missing_deps:
            print(f"  - Missing dependencies: {', '.join(missing_deps)}")
        if disk_info["free_gb"] <= 10:
            print("  - Insufficient disk space (need >10 GB free)")
    
    print("=" * 60)
    return 0 if ready else 1

if __name__ == "__main__":
    sys.exit(main())
