from aws_cdk import (
    Duration,
    Stack,
    aws_lambda as lambda_,
    CfnOutput,
    aws_iam as iam,
)
from aws_cdk.aws_apigatewayv2_alpha import (
    CorsHttpMethod,
    CorsPreflightOptions,
    HttpApi,
    HttpMethod
)
from aws_cdk.aws_apigatewayv2_integrations_alpha import HttpLambdaIntegration
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

        # Create HttpAPI
        api = HttpApi(
            self, "HttpAPI",
            cors_preflight=CorsPreflightOptions(
                allow_methods=[
                    CorsHttpMethod.DELETE,
                    CorsHttpMethod.GET,
                    CorsHttpMethod.OPTIONS,
                    CorsHttpMethod.POST,
                    CorsHttpMethod.PUT,
                ],
                allow_headers=[
                    "Authorization",
                    "Content-Type"
                ],
                allow_origins=["*"]
            )
        )

        # Integrate API into Lambda
        integration = HttpLambdaIntegration(
            "LambdaIntegration", docker_func
        )

        api.add_routes(
            path="/{proxy+}",
            methods=[
                HttpMethod.GET,
                HttpMethod.POST,
                HttpMethod.PUT,
                HttpMethod.PATCH,
                HttpMethod.DELETE,
            ],
            integration=integration,
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

        CfnOutput(
            self,
            "HttpApiUrl",
            value=api.url or "No URL found"
        )
