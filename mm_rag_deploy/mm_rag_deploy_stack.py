from aws_cdk import (
    Duration,
    Stack,
    aws_lambda as lambda_,
    CfnOutput,
    aws_iam as iam
)
from constructs import Construct

class MmRagDeployStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        docker_func = lambda_.DockerImageFunction(
            self, "FastAPIDockerDeploy",
            code=lambda_.DockerImageCode.from_image_asset("./multimodal-rag"),
            timeout=Duration.seconds(120),
            memory_size=1024
        )

        # Secrets Manager Access
        docker_func.add_to_role_policy(
            iam.PolicyStatement(
                actions=["secretsmanager:GetSecretValue"],
                resources=['arn:aws:secretsmanager:eu-central-1:156041399509:secret:prod/mm-rag/api_keys*']
            )
        )

        # Bedrock Runtime - run inference on 'amazon.titan-embed-image-v1'
        docker_func.add_to_role_policy(
            iam.PolicyStatement(
                actions=['bedrock:InvokeModel'],
                resources=["arn:aws:bedrock:eu-central-1::foundation-model/amazon.titan-embed-image-v1"]
            )
        )

        # S3 permissions
        docker_func.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:CreateBucket",
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:GetObjectAcl",
                    "s3:PutObjectAcl"
                ],
                resources=[
                    'arn:aws:s3:::mm-rag-bucket-may-hot',
                    'arn:aws:s3:::mm-rag-bucket-may-hot/*'
                ]
            )
        )

        # Dynamo DB permissions
        docker_func.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:CreateTable",
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:Query",
                    "dynamodb:Scan"
                ],
                resources=[
                    'arn:aws:dynamodb:eu-central-1:156041399509:table/files',
                    'arn:aws:dynamodb:eu-central-1:156041399509:table/users',
                    'arn:aws:dynamodb:eu-central-1:156041399509:table/users/index/PAT-gsi-index'
                ]
            )
        )

        docker_func_url = docker_func.add_function_url(
            auth_type=lambda_.FunctionUrlAuthType.NONE,
            cors={
                "allowed_methods": [lambda_.HttpMethod.ALL],
                "allowed_headers": ["*"],
                "allowed_origins": ["*"]
            }
        )

        CfnOutput(
            self,
            "FunctionUrlValue",
            value=docker_func_url.url
        )
