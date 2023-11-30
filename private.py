import os

HOME_DIR = "/Volumes/Pegasus32/data/NTT_Data/"
JAPAN_MESH4 = "/Volumes/Pegasus32/japan/mesh/japan_mesh4_CRS84.geojson"

def update_server():
    """
    To update and sync AWS and Pegasus32

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    cwd = os.getcwd()
    server = "/Volumes/Pegasus32/data/AWS_NTT_Data"
    os.chdir(server)
    command = "aws s3 sync s3://client-rt2-01h-6-tohokuuniversity2023/realtime ."
    os.system(command=command)
    os.chdir(cwd)
    print("Data downloaded")
    return