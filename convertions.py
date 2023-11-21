import subprocess

if __name__ == "__main__":
    odom_process = subprocess.Popen(['python', 'convert_odom.py'])
    elevation_process = subprocess.Popen(['python', 'convert_elevation.py'])
    cmd_vel_process = subprocess.Popen(['python', 'convert_cmd_vel.py'])

    try:
        odom_process.wait()
        elevation_process.wait()
        cmd_vel_process.wait()
    except KeyboardInterrupt:
        odom_process.terminate()
        elevation_process.terminate()
        cmd_vel_process.terminate()
