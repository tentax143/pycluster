import argparse
from winray.head import run_head
from winray.worker import run_worker

def main():
    parser = argparse.ArgumentParser(description="WinRay Distributed Task CLI")
    subparsers = parser.add_subparsers(dest="command")

    head_parser = subparsers.add_parser("head", help="Run the head node server")

    worker_parser = subparsers.add_parser("worker", help="Join as a worker node")
    worker_parser.add_argument("--head-ip", "-H", required=True, help="IP address of the head node")

    args = parser.parse_args()

    if args.command == "head":
        run_head()
    elif args.command == "worker":
        run_worker(args.head_ip)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
