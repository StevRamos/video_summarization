import os
from tempfile import NamedTemporaryFile

MIME_TYPES = {
    'csv': 'text/csv',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'ppt': 'application/vnd.ms-powerpoint',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'pdf': 'application/pdf',
    'doc': 'application/msword',
    'gz': 'application/gzip',
    'gif': 'image/gif',
    'ico': 'image/vnd.microsoft.icon',
    'jar': 'application/java-archive',
    'jpeg': 'image/jpeg',
    'jpg': 'image/jpeg',
    'htm': 'text/html',
    'html': 'text/html',
    'json': 'application/json',
    'mp3': 'audio/mpeg',
    'mpeg': 'video/mpeg',
    'odp': 'application/vnd.oasis.opendocument.presentation',
    'ods': 'application/vnd.oasis.opendocument.spreadsheet',
    'odt': 'application/vnd.oasis.opendocument.text',
    'png': 'image/png',
    'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'rar': 'application/vnd.rar',
    'svg': 'image/svg+xml',
    'tar': 'application/x-tar',
    'txt': 'text/plain',
    'vsd': 'application/vnd.visio',
    'wav': 'audio/wav',
    'xhtml': 'application/xhtml+xml',
    'xls': 'application/vnd.ms-excel',
    'xml': 'application/xml if not readable from casual users (RFC 3023, section 3) text/xml if readable from casual users (RFC 3023, section 3)',
    'zip': 'application/zip',
    '7z': 'application/x-7z-compressed',
    'mp4': 'video/mp4',
    'm1v': 'video/mpeg',
    'webm': 'video/webm',
    'm4v': 'video/x-m4v',
    'wmv': 'video/x-ms-wmv',
    'avi': 'video/x-msvideo',
    'mpg': 'video/mpeg',
    'flv': 'video/x-flv',
    'pk': 'application/octet-stream'
}


def get_mime_types(file_names):
    mime_types = []
    for file in file_names:
        extension = file.split(".")[-1]
        mime_types.append(MIME_TYPES[extension])

    return mime_types

def save_file(file, path):
    if not os.path.exists(path):
        os.makedirs(path)

    file.save(path + file.filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in MIME_TYPES.keys()

