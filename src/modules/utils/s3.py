"""Functions for retrieving from or uploading data to S3."""
import json
import boto3

# Create s3 client
s3 = boto3.client("s3")


def get_from_s3(bucket: str, key: str, decode: bool = True) -> str:
    """Get and decode response body from S3 object.
    
    Parameters
    ----------
    bucket : str
        S3 bucket name
    key: str
        S3 key
    decode: bool
        True if response should be decoded

    Returns
    -------
    str
        decoded response
        
    Raises
    ------
    AssertionError
        Unsuccessful retrieval from S3
    """
    try:
        s3_object = s3.get_object(Bucket=bucket, Key=key)
        assert s3_object["ResponseMetadata"]["HTTPStatusCode"] == 200
        response = s3_object["Body"].read()
        if decode:
            return response.decode()
        else:
            return response

    except AssertionError:
        print("Cannot retrieve from location: s3://{}/{}".format(bucket, key))
        raise AssertionError(s3_object["ResponseMetadata"])


def put_to_s3(file: str, bucket: str, key: str):
    """Encode and write file to S3.
    
    Parameters
    ----------
    file: str
        file in string format, example: json.dumps(), pd.to_csv()
    bucket : str
        S3 bucket name
    key: str
        S3 key
    """
    file_bytes = bytes(file.encode("UTF-8"))
    s3.put_object(Body=file_bytes, Bucket=bucket, Key=key)
    print("Uploaded file to s3://{}/{}".format(bucket, key))


def list_s3(bucket: str, path: str) -> list:
    """List objects in S3.
    
    Parameters
    ----------
    bucket : str
        S3 bucket name
    path: str
        file path
    
    Returns
    -------
    list
        list of object names
    """
    raw_objects = s3.list_objects(Bucket=bucket, Prefix=path)["Contents"]
    return list(map(lambda x: x["Key"], raw_objects))
