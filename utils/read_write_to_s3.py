import pandas as pd
import boto3
import io
from io import StringIO, BytesIO
import pickle
import s3fs
import numpy as np
import json
import joblib
import csv
import os
import shutil


def write_pd_to_s3(out_df, bucket_name, full_path, index=True, header=True, quoting=None, sep=","):
    """
    Write a pandas DataFrame to S3 as CSV **without needing to save it locally**.

    Args:
        out_df (pd.Dataframe): dataframe to upload
        bucket_name (str): S3 bucket name
        full_path (str): full object key (folder/filename.csv)
        index (bool): include index
        header (bool): include header
        quoting: csv quoting option
        sep (str): delimiter

    Example of bucket_name: ctg-ctgdl-prod-role-workspace
    Example of full_path: hr-hc-view/data/WPA_and_OA/test_df.csv
                         (No leading "/" required; must end with .csv)

    Reference:
    https://stackoverflow.com/questions/38154040/save-dataframe-to-csv-directly-to-s3-python
    """
    csv_buffer = StringIO()
    out_df.to_csv(csv_buffer, index=index, header=header, quoting=quoting, sep=sep)

    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name, full_path).put(Body=csv_buffer.getvalue())


def read_csv_from_s3(bucket_name, object_key, **kwargs):
    """
    Read a CSV file from S3 into a pandas DataFrame.

    Args:
        bucket_name (str)
        object_key (str): full S3 key (including path + .csv)

    kwargs: passed to pandas.read_csv(), e.g., `usecols`, `header`, etc.

    Returns:
        pd.DataFrame
    """
    client = boto3.client('s3')
    obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = obj['Body']
    csv_string = body.read().decode('utf-8')

    return pd.read_csv(StringIO(csv_string), **kwargs)

def read_files_loop(bucket_name, prefix, **kwargs):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    prefix_objs = bucket.objects.filter(Prefix=prefix)
    dfs = []

    for obj in prefix_objs:
        key = obj.key
        df = read_csv_from_s3(bucket_name, key, **kwargs)
        dfs.append(df)

    print("{0} files read".format(len(dfs)))
    return dfs


# function to save model in S3 as pickle file
def write_pickle_file(model, bucket_name, key):
    """
    This function writes a pickle file to S3

    Parameters:
        model: model object
        bucket_name (string): name of S3 bucket
        key (string): path to save model object
    """
    s3 = s3fs.S3FileSystem()

    with s3.open("s3://" + bucket_name + "/" + key, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Object saved here: " + "s3://" + bucket_name + "/" + key)


# function to load pickle file model from S3
def load_pickle_file(bucket_name, key):
    """
    This function loads a model object from S3.

    Parameters:
        bucket_name (string): name of S3 bucket
        key (string): path to retrieve model object

    Return:
        model: model object
    """
    s3 = s3fs.S3FileSystem()

    with s3.open("s3://" + bucket_name + "/" + key, "rb") as f:
        model = pickle.load(f)

    return model

# write dictionary to s3
def write_dict_s3(dictionary, bucket, key, orient='columns'):
    """
    This function writes a dictionary to S3.

    Parameters:
        dictionary (dict): dictionary object
        bucket (string): name of S3 bucket
        key (string): path to save dictionary
        orient (string): {'columns', 'index', 'tight'}, default 'columns'
    """
    df = pd.DataFrame.from_dict(dictionary, orient=orient)
    write_pd_to_s3(df, bucket, key)
    print("Written to: " + key)


# write out to text file in s3
def write_txt_to_s3(bucket, path, output_list):
    """
    This function writes a text file to S3.

    Parameters:
        bucket (string): name of S3 bucket
        path (string): path to save text file
        output_list (list): list of items to write to text file
    """
    s3r = boto3.resource('s3')
    output = s3r.Object(bucket, path)
    output_text = ""

    for item in output_list:
        if isinstance(item, list):
            output_text = output_text + ','.join(item) + '\n\n'
        elif isinstance(item, pd.core.frame.DataFrame):
            output_text = output_text + item.to_string() + '\n\n'
        elif isinstance(item, dict):
            output_text = output_text + str(list(item.items())) + '\n\n'
        elif isinstance(item, np.ndarray):
            output_text = output_text + str(list(item)) + '\n\n'
        else:
            output_text = output_text + str(item) + '\n\n'

    output.put(Body=output_text)
    print('Text file written to' + path)

# function to load json file model from S3
def load_json_file(bucket_name, key):
    """
    This function loads a json from S3.

    Parameters:
        bucket_name (string): name of S3 bucket
        key (string): path to retrieve json file

    Return
        file: json file as dict
    """
    s3_resource = boto3.resource('s3')
    content_object = s3_resource.Object(bucket_name, key)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    file = json.loads(file_content)
    return file


# function to write json file model to S3
def write_json_file(bucket_name, key, dictionary):
    """
    This function writes a json to S3.

    Parameters:
        bucket_name (string): name of S3 bucket
        key (string): path to retrieve json file
        dictionary (dict): dict object that will be stored in json file
    """
    s3_resource = boto3.resource('s3')
    s3_object = s3_resource.Object(bucket_name, key)
    s3_object.put(Body=json.dumps(dictionary, indent=4))
    print("JSON file written to " + key)

# function to get the S3 subfolders in a sorted list
def get_s3_folders(bucket_name, prefix):
    """
    This function retrieves the S3 subfolders in a sorted list.

    Parameters:
        bucket_name (string): name of S3 bucket
        prefix (string): prefix for S3 folders

    Return:
        list_prefix: list of folders in alphabetical order
    """
    client = boto3.client('s3')
    result = client.list_objects(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    list_prefix = []

    for o in result.get('CommonPrefixes'):
        list_prefix.append(o.get('Prefix'))

    list_prefix.sort()
    return list_prefix


# function to write joblib file to S3
def write_joblib_file(file, bucket_name, key):
    """
    This function writes out an object to a joblib file.

    Parameters:
        bucket_name (string): name of S3 bucket
        key (string): path to saved joblib
        file (object): object to be saved as joblib
    """
    s3_client = boto3.client('s3')

    with BytesIO() as f:
        joblib.dump(file, f)
        f.seek(0)
        s3_client.put_object(Body=f.read(), Bucket=bucket_name, Key=key)

    print("Joblib file written to " + key)


# function to load joblib file from S3
def load_joblib_file(bucket_name, key):
    """
    This function retrieves a joblib file from S3

    Parameters:
        bucket_name (string): name of S3 bucket
        key (string): path to saved joblib

    Return
        file: joblib file
    """
    s3_resource = boto3.resource('s3')
    with BytesIO() as data:
        s3_resource.Bucket(bucket_name).download_fileobj(key, data)
        data.seek(0)
        file = joblib.load(data)
    return file


# function to save printed graph to S3
def save_plot_s3(bucket_name, key, figure):
    """
    This function saves an image file from S3

    Parameters:
        bucket_name (string): name of S3 bucket
        key (string): path to saved joblib
        figure (figure): name of figure saved as plt.gcf()
    """
    img_data = BytesIO()
    figure.savefig(img_data, format='jpg', bbox_inches='tight')
    img_data.seek(0)

    s3_s3fs = s3fs.S3FileSystem(anon=False) # Uses default credentials
    with s3_s3fs.open("s3://" + bucket_name + "/" + key, "wb") as f:
        f.write(img_data.getbuffer())

def upload_file_to_s3(file, file_name, relative_path, s3_client, s3_bucket, s3_folder_path):
    """Upload a file object directly to S3"""

    try:
        # Download file content (read from file-like object)
        file_content = file.read()

        # Create S3 key with folder structure
        if relative_path:
            s3_file_key = f"{s3_folder_path.rstrip('/')}/{relative_path}/{file_name}"
        else:
            s3_file_key = f"{s3_folder_path.rstrip('/')}/{file_name}"

        # Upload to S3
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_file_key,
            Body=io.BytesIO(file_content)
        )

        s3_location = f"s3://{s3_bucket}/{s3_file_key}"
        print(f"Uploaded file to S3: {s3_location}")

        return s3_location

    except Exception as e:
        print(f"Error uploading file to S3: {file_name}, Error: {str(e)}")
        return "Upload failed"

def upload_metadata_to_s3(all_files, s3_client, s3_bucket, s3_folder_path):
    """Upload metadata CSV directly to S3"""

    if not all_files:
        print("No files to add to metadata CSV")
        return

    try:
        # Create CSV in memory
        csv_buffer = io.StringIO()
        fieldnames = all_files[0].keys()
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_files)

        # Upload CSV to S3
        csv_s3_key = f"{s3_folder_path.rstrip('/')}/checkpoint_files.csv"

        s3_client.put_object(
            Bucket=s3_bucket,
            Key=csv_s3_key,
            Body=csv_buffer.getvalue().encode('utf-8')
        )

        print(f"CSV file uploaded to S3: s3://{s3_bucket}/{csv_s3_key}")

    except Exception as e:
        print(f"Error uploading CSV to S3: {str(e)}")

def save_faiss_index_to_s3(faiss_index, bucket_name, key_prefix):
    """
    Save a FAISS index to S3.

    Args:
        faiss_index: The FAISS index object from langchain
        bucket_name (str): S3 bucket name
        key_prefix (str): S3 key prefix (path)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        s3_client = boto3.client("s3")

        # Create a temporary directory to save FAISS files locally
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        temp_dir = os.path.join(project_root, "tmp")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save FAISS index locally (LangChain stores 2 files: index.faiss + index.pkl)
            local_index_path = os.path.join(temp_dir, "index")
            faiss_index.save_local(local_index_path)

            # upload the index files to S3
            index_file = os.path.join(local_index_path, "index.faiss")
            docstore_file = os.path.join(local_index_path, "index.pkl")

            # Upload FAISS binary index
            s3_client.upload_file(
                index_file,
                bucket_name,
                f"{key_prefix}/index.faiss"
            )

            # Upload FAISS metadata (docstore.pkl)
            s3_client.upload_file(
                docstore_file,
                bucket_name,
                f"{key_prefix}/index.pkl"
            )

            print(f"FAISS index uploaded to S3:")
            print(f"   s3://{bucket_name}/{key_prefix}/index.faiss")
            print(f"   s3://{bucket_name}/{key_prefix}/index.pkl")
            return True

        except Exception as e:
            print(f"❌ Error saving FAISS locally or uploading to S3: {str(e)}")
            return False
        
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"Error saving FAISS index to S3: {str(e)}")
        return False

def load_faiss_index_from_s3(bucket_name, key_prefix, embeddings):
    """
    Load a FAISS index from S3.

    Args:
        bucket_name (str): S3 bucket name
        key_prefix (str): S3 key prefix (path)
        embeddings: The embedding function to use with the FAISS index

    Returns:
        FAISS: The loaded FAISS index, or None if loading failed
    """

    from langchain.vectorstores import FAISS
    import boto3, os

    try:
        s3_client = boto3.client("s3")

        # Create a temporary directory at the project level
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        temp_dir = os.path.join(project_root, "tmp")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Create directory for index files inside temp folder
            local_index_path = os.path.join(temp_dir, "index")
            os.makedirs(local_index_path, exist_ok=True)

            # === Download from S3 ===
            # Download binary index
            s3_client.download_file(
                bucket_name,
                f"{key_prefix}/index.faiss",
                os.path.join(local_index_path, "index.faiss")
            )

            # Download metadata store (document chunks etc.)
            s3_client.download_file(
                bucket_name,
                f"{key_prefix}/index.pkl",
                os.path.join(local_index_path, "index.pkl")
            )

            # === Load back into FAISS index ===
            faiss_index = FAISS.load_local(
                local_index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            print(f"✅ FAISS index loaded from S3:")
            print(f"   s3://{bucket_name}/{key_prefix}/index.faiss")
            print(f"   s3://{bucket_name}/{key_prefix}/index.pkl")

            return faiss_index

        except Exception as e:
            print(f"Error loading FAISS index from S3: {str(e)}")
            return None

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    except Exception as e:
        print(f"Failed to initialize S3 client: {str(e)}")
        return None

if __name__ == '__main__':
    bucket = ''
    full_path = ''
    dataframe = read_files_loop(bucket, full_path, 14)
    write_pd_to_s3(dataframe[0], bucket, full_path)