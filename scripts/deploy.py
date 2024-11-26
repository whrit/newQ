# scripts/deploy.py
import boto3
import yaml
import os
import time
from botocore.exceptions import ClientError

class CloudDeployer:
    def __init__(self):
        self.ecs = boto3.client('ecs')
        self.cloudwatch = boto3.client('cloudwatch')
        self.logs = boto3.client('logs')
        
        # Load configuration
        with open('config/cloud_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
    
    def create_log_group(self):
        """Create CloudWatch log group if it doesn't exist"""
        try:
            self.logs.create_log_group(
                logGroupName=self.config['monitoring']['cloudwatch']['log_group']
            )
            print(f"Created log group: {self.config['monitoring']['cloudwatch']['log_group']}")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                raise
    
    def create_ecs_cluster(self):
        """Create ECS cluster if it doesn't exist"""
        try:
            self.ecs.create_cluster(
                clusterName=self.config['aws']['ecs']['cluster'],
                capacityProviders=['FARGATE'],
                defaultCapacityProviderStrategy=[
                    {
                        'capacityProvider': 'FARGATE',
                        'weight': 1
                    }
                ]
            )
            print(f"Created ECS cluster: {self.config['aws']['ecs']['cluster']}")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ClusterAlreadyExistsException':
                raise
    
    def register_task_definition(self):
        """Register ECS task definition"""
        response = self.ecs.register_task_definition(
            family=self.config['aws']['ecs']['task_definition'],
            networkMode='awsvpc',
            requiresCompatibilities=['FARGATE'],
            cpu=str(self.config['aws']['ecs']['container_cpu']),
            memory=str(self.config['aws']['ecs']['container_memory']),
            containerDefinitions=[
                {
                    'name': 'trading-bot',
                    'image': f"{os.getenv('ECR_REPOSITORY')}:latest",
                    'essential': True,
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': self.config['monitoring']['cloudwatch']['log_group'],
                            'awslogs-region': self.config['aws']['region'],
                            'awslogs-stream-prefix': 'ecs'
                        }
                    },
                    'environment': [
                        {
                            'name': 'TRADING_MODE',
                            'value': os.getenv('TRADING_MODE', 'paper')
                        }
                    ],
                    'secrets': [
                        {
                            'name': 'ALPACA_API_KEY',
                            'valueFrom': os.getenv('ALPACA_API_KEY_ARN')
                        },
                        {
                            'name': 'ALPACA_SECRET_KEY',
                            'valueFrom': os.getenv('ALPACA_SECRET_KEY_ARN')
                        }
                    ]
                }
            ]
        )
        return response['taskDefinition']['taskDefinitionArn']
    
    def create_or_update_service(self, task_definition_arn):
        """Create or update ECS service"""
        try:
            self.ecs.create_service(
                cluster=self.config['aws']['ecs']['cluster'],
                serviceName=self.config['aws']['ecs']['service'],
                taskDefinition=task_definition_arn,
                desiredCount=self.config['aws']['ecs']['desired_count'],
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': os.getenv('SUBNET_IDS').split(','),
                        'securityGroups': [os.getenv('SECURITY_GROUP_ID')],
                        'assignPublicIp': 'ENABLED'
                    }
                }
            )
            print(f"Created ECS service: {self.config['aws']['ecs']['service']}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ServiceAlreadyExists':
                self.ecs.update_service(
                    cluster=self.config['aws']['ecs']['cluster'],
                    service=self.config['aws']['ecs']['service'],
                    taskDefinition=task_definition_arn,
                    desiredCount=self.config['aws']['ecs']['desired_count']
                )
                print(f"Updated ECS service: {self.config['aws']['ecs']['service']}")
            else:
                raise

    def setup_monitoring(self):
        """Set up CloudWatch alarms"""
        # CPU Utilization Alarm
        self.cloudwatch.put_metric_alarm(
            AlarmName=f"{self.config['aws']['ecs']['service']}-cpu-utilization",
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='CPUUtilization',
            Namespace='AWS/ECS',
            Period=300,
            Statistic='Average',
            Threshold=self.config['monitoring']['cloudwatch']['alerts']['cpu_threshold'],
            AlarmDescription='CPU utilization is too high',
            Dimensions=[
                {
                    'Name': 'ClusterName',
                    'Value': self.config['aws']['ecs']['cluster']
                },
                {
                    'Name': 'ServiceName',
                    'Value': self.config['aws']['ecs']['service']
                }
            ]
        )

    def deploy(self):
        """Main deployment method"""
        print("Starting deployment...")
        
        # Create resources
        self.create_log_group()
        self.create_ecs_cluster()
        
        # Register task definition
        task_definition_arn = self.register_task_definition()
        
        # Create/update service
        self.create_or_update_service(task_definition_arn)
        
        # Setup monitoring
        self.setup_monitoring()
        
        print("Deployment completed successfully!")

if __name__ == "__main__":
    deployer = CloudDeployer()
    deployer.deploy()