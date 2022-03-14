import os
import subprocess

from .dashboard_server import DashboardServer


class Dashboard:
    def __init__(self, tasks):
        self.tasks = tasks
        
    def run(self):
        # Run dashboard UI
        subprocess.Popen(["streamlit", "run", os.path.join('koria', 'dashboard', "dashboard_ui.py")])
        
        # Run dashboard server
        server = DashboardServer(tasks=self.tasks)
        server.run()

        while True:
            pass
