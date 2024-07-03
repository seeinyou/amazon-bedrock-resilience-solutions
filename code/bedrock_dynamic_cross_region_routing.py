"""
Amazon Bedrock Cross-Region Resilience Solution
This example script implements the Amazon Bedrock Cross-Region Resilience Solution using the AWS Python SDK (boto3).

Key configurations:
    MAX_RETRY_TIME (int) -- The maximum number of retry attempts for Bedrock APIs. Recommended values are 3-5.
    MULTI_REGION_RETRY (bool) -- Retry Bedrock APIs in multiple regions if APIs are temporarily unable to respond correctly.
    MAX_RETRY_TIMES_FOR_EACH_REGION (int) -- The maximum number of retry attempts for each region before trying a different region.
        The total max_retry_times_for_each_region cannot exceed max_retry_time.
    NEXT_RETRY_TIME_WINDOW (int) -- The time window (in seconds) added to the current time to set the next available time for failed endpoints.
"""
import json
import time
import os
import fcntl

import boto3
import botocore

# Global configurations
MAX_RETRY_TIME = 5
NEXT_RETRY_TIME_WINDOW = 3600  # 1 hour
MULTI_REGION_RETRY = True
MAX_RETRY_TIMES_FOR_EACH_REGION = 2
TIMEOUT = 5000  # Milliseconds

failed_regions = []

if MULTI_REGION_RETRY:
    config_retry_times = 0
else:
    MAX_RETRY_TIMES_FOR_EACH_REGION = MAX_RETRY_TIME

config = botocore.config.Config(
    read_timeout=900,
    connect_timeout=900,
    retries={"max_attempts": 0}
)


def get_validate_regions_from_conf(region_configs):
    """
    Filter the list of endpoints to keep only the available endpoints in descending order
    based on each endpoint's next available time.

    Args:
        region_configs (list) -- The list of endpoints, loaded from the "bedrock_endpoints"
    """
    validate_regions = []

    for regional_conf in region_configs:
        print('### REGION CONF:', regional_conf)

        if regional_conf['next_available_time'] <= current_time:
            validate_regions.append(regional_conf['region'])

    return validate_regions


def bedrock_invoke_model_message_with_retry(request_data, model_id, validate_regions, max_retry_time):
    """
    Send a request to Amazon Bedrock InvokeModel API and retry according to related configurations
    when API request does not return a valid response.
    
    Args:
        request_data (dict): Data for Bedrock API request.
        model_id (str): Requested Bedrock model Id.
        validate_regions (list): The list of available endpoints.
        max_retry_time (int): The maximum number of retry attempts for Bedrock APIs.
    """
    if len(validate_regions) == 0:
        return False

    retry_time = 0

    # Bedrock API parameters
    accept = 'application/json'
    content_type = 'application/json'
    
    for region_name in validate_regions: # Loop through available endpoints until configured exit criteria is met.

        if region_name is not None and retry_time < max_retry_time:
            bedrock = boto3.client('bedrock-runtime', region_name=region_name, config=config) # Initialize a regional Bedrock client

            for one_region_retry_time in range(MAX_RETRY_TIMES_FOR_EACH_REGION):
                # Retry the request to one region

                if retry_time < max_retry_time:

                    try:
                        response = bedrock.invoke_model(body=request_data, modelId=model_id, accept=accept, contentType=content_type)

                        if response:
                            return response
                        else:
                            if one_region_retry_time == 0:
                                failed_regions.append(region_name) # Add the region to the failed region list

                            retry_time += 1
                            continue

                    except (botocore.exceptions.ClientError, Exception) as e:
                        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
                        
                        if one_region_retry_time == 0:
                            failed_regions.append(region_name) # Add the region to the failed region list

                        retry_time += 1
                        continue
                else:
                    break
        else:
            break

    return False


def disable_region_in_conf(raw_region_configs, disable_regions):
    """
    Set the next available timestamp to disable a region in the configuration file

    Args:
        raw_region_configs (dict): The raw list of Bedrock endpoints from the "bedrock_endpoints" file.
        disable_regions (list): The endpoints to be calculated and set next available time.
    """
    # Calcuate failed endpoints' next available time
    next_timestamp = current_time + NEXT_RETRY_TIME_WINDOW
    print('### NEXT TIME:', next_timestamp)

    disable_count = len(disable_regions)

    for region_data in raw_region_configs:
        if region_data['region'] in disable_regions:
            region_data['next_available_time'] = next_timestamp

    return raw_region_configs


def write_json_to_file_with_lock(file_path, data):
    """
    Overwrite the Bedrock endpoint configuration file with file locking to handle concurrent writes.

    Args:
        file_path (str): The path to the Bedrock endpoint configuration file where the JSON data will be written.
        data (dict): The JSON data to write to the file.
    """
    # Convert the Python dictionary to a JSON string
    json_data = json.dumps(data)

    try:
        with open(file_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Acquire an exclusive lock on the file
            f.write(json_data)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file

        return True

    except IOError as e:
        print(f"Error writing to file: {e}")
        return False


# An example of the calling Bedrock API workflow
# Initiates Sending an Amazon Bedrock API Request
## Get current timestamp
current_time = round(time.time())
print('# TIME:', current_time)

## Load the Bedrock Endpoint Configuration File
filename = 'bedrock_endpoints.conf'

with open(filename) as f:
    endpoint_config = f.read()

raw_region_configs = json.loads(endpoint_config)
print('# CONFIG:', raw_region_configs)

## Retrieve currently validate regions from the bedrock_endpoints.conf
validate_regions = get_validate_regions_from_conf(raw_region_configs)
print('# VALIDATE REGIONS:', validate_regions)

## Construct a Bedrock InvokeModel API request
system_prompt = ''

prompt = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Say Hello",
            }
        ]
    }
]

request_body = {"messages": prompt,
    "system": system_prompt,
    "max_tokens": 2000,
    "temperature":0.01,
    "top_k":250,
    "top_p":0.5,
    "anthropic_version": "bedrock-2023-05-31",
    "stop_sequences": ["\n\nHuman:", "</response>", "</result>"]
}

body = json.dumps(request_body)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

## Send the Request
response = bedrock_invoke_model_message_with_retry(body, model_id, validate_regions, MAX_RETRY_TIME)
print('# BEDROCK RESPONSE:', response)


if response:
    output = json.loads(response.get('body').read())
    print('# BEDROCK OUTPUT:', output['content'][0]['text'])

## Check for Failed Endpoints
if len(failed_regions) > 0:
    ## Update Bedrock Endpoint Configuration File
    new_config = disable_region_in_conf(raw_region_configs, failed_regions)
    
    ### Write the updated region configurations back to the bedrock_region_configs.conf
    conf_save_result = write_json_to_file_with_lock(filename, new_config)

else:
    print('No need to update bedrock_region_configs.conf')
