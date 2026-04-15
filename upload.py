import boto3
import os

def main():
    os.makedirs('files', exist_ok=True)

    BUCKET_NAME = 'hockey.josephbensenportfolio.com'
    s3 = boto3.client('s3')

    # Local parquet file
    files = [path for path in os.listdir('files') if ".csv" in path]
    for file in files:
        local_file = os.path.join(os.path.curdir, 'files', file)
        s3_key = f'CSVs/{file}'
        s3.upload_file(Filename=local_file, Bucket=BUCKET_NAME, Key=s3_key)

    print("CSVs uploaded!")

if __name__ == '__main__':
    main()