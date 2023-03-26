"""
download the data from Google Drive

file_id: 1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg
file_name: iu_xray.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg" -O iu_xray.zip && rm -rf /tmp/cookies.txt

file_id: 1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E
file_name: mimic_cxr.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E" -O mimic_cxr.zip && rm -rf /tmp/cookies.txt
"""
from google_drive_downloader import GoogleDriveDownloader as gdd


def download(name):

    if name == 'iu_xray':
        gdd.download_file_from_google_drive(file_id='1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg',
                                            dest_path='data/iu_xray.zip')
    elif name == 'mimic_cxr':
        gdd.download_file_from_google_drive(file_id='1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E',
                                            dest_path='data/mimic_cxr.zip')


download('mimic_cxr')
