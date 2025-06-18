import aws_cdk as core
import aws_cdk.assertions as assertions

from mm_rag_deploy.mm_rag_deploy_stack import MmRagDeployStack

# example tests. To run these tests, uncomment this file along with the example
# resource in infrastructure/mm_rag_deploy_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = MmRagDeployStack(app, "mm-rag-deploy")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
