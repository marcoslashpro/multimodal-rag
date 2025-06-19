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
import os
from pathlib import Path

class MmRagDeployStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create a dedicated IAM role for the Lambda function.
        lambda_role = iam.Role(
            self, "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for the Docker-based Lambda function",
        )

        # Option A: Attach an existing managed policy by ARN
        custom_policy = iam.ManagedPolicy.from_managed_policy_arn(
            self,
            "CustomPolicy",
            "arn:aws:iam::156041399509:policy/mm-rag-policy"
        )
        lambda_role.add_managed_policy(custom_policy)

        docker_func = lambda_.DockerImageFunction(
            self, "FastAPIDockerDeploy",
            code=lambda_.DockerImageCode.from_image_asset(
                directory=f'{Path(__file__).parent.parent}/.docker/',
                file='Dockerfile'
            ),
            timeout=Duration.seconds(120),
            memory_size=1024,
            role=lambda_role,
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

        CfnOutput(
            self,
            "HttpApiUrl",
            value=api.url or "No URL found"
        )
