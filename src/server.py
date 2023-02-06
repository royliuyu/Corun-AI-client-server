from PIL import Image
import io
import socket
import infer
from util import str2dict


ip , port = '127.0.0.1', 8000
def work():
    s = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, port))
    s.listen(3)
    while True:
        conn, addr = s.accept()
        print(f'Received request from {addr} !')
        header = conn.recv(1024)
        print('header recv:', header)
        try:
            format, args_str = header.decode().split('|')
            args = str2dict(args_str)
            conn.send(b'ok')
        except:
            print('Fail to connect !')
            conn.close()
            continue
        while True:
            file_info = conn.recv(1024)
            data_len, file_name = file_info.decode().split('|')
            args['file_name'] = file_name
            conn.send(b'ok')
            if data_len and file_name:
                print(file_name)
                # newfile = open(os.path.join(dir, file_name), 'wb') # save file transfered from server
                file = b''
                data_len = int(data_len)
                get = 0
                while get < data_len: # recieve data
                    data = conn.recv(data_len//2)
                    file += data  ## binary code
                    get += len(data)
                conn.send(b'ok')
                print(f' {data_len} bytes to transfer, recieved {len(file)} bytes.')
                if file:  # file is in format of binary byte
                    pass
                    # newfile.write(file[:])  # save file transfered from server
                    # newfile.close()

                    # image = np.fromfile(file, dtype=np.uint8)  # doesn't work
                    image = Image.open(io.BytesIO(file))  # convert binary bytes to PIL image
                    prd, latency = infer.work(image, args)
                    print(prd, latency)

            reply = conn.recv(1024)
            print('rely recv: ', reply)
            if reply.decode() == 'done':
                break
work()