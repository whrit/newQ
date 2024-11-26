# scripts/monitor.py
import boto3
import time
from datetime import datetime, timedelta

class CloudMonitor:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.logs = boto3.client('logs')
        
    def get_metrics(self, cluster_name, service_name):
        """Get ECS service metrics"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        response = self.cloudwatch.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': 'cpu',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/ECS',
                            'MetricName': 'CPUUtilization',
                            'Dimensions': [
                                {
                                    'Name': 'ClusterName',
                                    'Value': cluster_name
                                },
                                {
                                    'Name': 'ServiceName',
                                    'Value': service_name
                                }
                            ]
                        },
                        'Period': 300,
                        'Stat': 'Average'
                    }
                },
                {
                    'Id': 'memory',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': 'AWS/ECS',
                            'MetricName': 'MemoryUtilization',
                            'Dimensions': [
                                {
                                    'Name': 'ClusterName',
                                    'Value': cluster_name
                                },
                                {
                                    'Name': 'ServiceName',
                                    'Value': service_name
                                }
                            ]
                        },
                        'Period': 300,
                        'Stat': 'Average'
                    }
                }
            ],
            StartTime=start_time,
            EndTime=end_time
        )
        
        return response['MetricDataResults']

    def get_recent_logs(self, log_group, log_stream_prefix):
        """Get recent CloudWatch logs"""
        streams = self.logs.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=log_stream_prefix,
            orderBy='LastEventTime',
            descending=True,
            limit=1
        )
        
        if not streams['logStreams']:
            return []
        
        logs = self.logs.get_log_events(
            logGroupName=log_group,
            logStreamName=streams['logStreams'][0]['logStreamName'],
            limit=100
        )
        
        return logs['events']

if __name__ == "__main__":
    monitor = CloudMonitor()
    
    while True:
        # Get metrics
        metrics = monitor.get_metrics('trading-bot-cluster', 'trading-bot-service')
        
        # Get recent logs
        logs = monitor.get_recent_logs('/ecs/trading-bot', 'ecs')
        
        # Print monitoring information
        print("\n=== Trading Bot Monitoring ===")
        print(f"Time: {datetime.utcnow().isoformat()}")
        print("\nMetrics:")
        for metric in metrics:
            if metric['Values']:
                print(f"{metric['Id']}: {metric['Values'][-1]:.2f}%")
        
        print("\nRecent Logs:")
        for log in logs[-5:]:
            print(f"{log['timestamp']}: {log['message']}")
        
        time.sleep(300)  # Update every 5 minutes