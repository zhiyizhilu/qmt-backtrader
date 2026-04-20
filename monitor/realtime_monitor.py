import time
import threading
from typing import Dict, List, Optional, Any
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

class RealtimeMonitor:
    """实时监控"""
    
    def __init__(self):
        """初始化监控"""
        self.data = {}
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.running = False
        self.thread = None
    
    def setup_layout(self):
        """设置布局"""
        self.app.layout = html.Div([
            html.H1('实时监控仪表盘'),
            dcc.Graph(id='equity-graph'),
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
        ])
        
        @self.app.callback(
            Output('equity-graph', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_graph(n):
            """更新图表"""
            if 'equity' in self.data:
                return {
                    'data': [{
                        'x': list(range(len(self.data['equity']))),
                        'y': self.data['equity'],
                        'type': 'line'
                    }],
                    'layout': {
                        'title': '实时净值曲线'
                    }
                }
            else:
                return {
                    'data': [],
                    'layout': {
                        'title': '实时净值曲线'
                    }
                }
    
    def update_data(self, key: str, value: Any):
        """更新数据"""
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
    
    def start(self):
        """启动监控"""
        self.running = True
        self.thread = threading.Thread(target=self.run_server)
        self.thread.daemon = True
        self.thread.start()
    
    def run_server(self):
        """运行服务器"""
        self.app.run_server(debug=False, port=8050)
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join()
