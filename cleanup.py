import os
import signal
import subprocess

def kill_process_on_port(port):
    try:
        # Get the PID of the process using the port
        result = subprocess.run(['lsof', '-i', f':{port}'], 
                               capture_output=True, text=True)
        
        # Parse the output to find PIDs
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:  # First line is header
            for line in lines[1:]:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    print(f"Killing process {pid} on port {port}")
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        print(f"Process {pid} not found")
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")

# Kill processes on the ports we'll be using
ports = [8000, 8001, 8002, 8003]
for port in ports:
    kill_process_on_port(port)

print("Cleanup complete")