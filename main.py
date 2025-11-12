#!/usr/bin/env python3

from common.modules import sys, traceback
from gui.experiment_launcher import ExperimentLauncher

def main():
    """Main Entry Point for Application"""
    try:
        launcher = ExperimentLauncher()
        launcher.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()