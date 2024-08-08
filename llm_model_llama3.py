import boto3
import json

prompt = '''
You are a weather analyst. You create a weekly weather report called WWR.
Instructions to generate WWR report - 
1) User to provide the location and time duration for the weekly report.
2) If user do not provide a time duration, take immediate last week as time duration.
3) Fetch weather report for that period
4) Show the result in a table
5) create a pivot showing different conditions and count of condition in a table
'''

bedrock=boto3.client(service_name='bedrock-runtime')

api_body={
    'prompt':prompt,
    'max_gen_len':512,
    'temperature':0.2,
    'top_p':0.9
}

body=json.dumps(api_body)

response = bedrock.invoke_model(
    modelId="meta.llama3-70b-instruct-v1:0",
    contentType= "application/json",
    accept= "application/json",
    body= body
)

response_body = json.loads(response.get("body").read())
response_text = response_body['generation']

print(response_text)

