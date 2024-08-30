import loader 
import detection
import socket
import importlib
import sys


if __name__ == '__main__':
    args = sys.argv[1:]
    use_case = args[0]

    loader_instance = loader.Loader(use_case)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('localhost', 10000)
    server_socket.bind(server_address)

    server_socket.listen(1)

    while True:
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()

        try:        
            data = connection.recv(1024)
            message = data.decode()
            if message != "":
                print(message)
            
            if "detection" in message:
                importlib.reload(detection)
                detection_instance = detection.Detection()
                detection_instance.set_loader(loader_instance)
                print("Detection completed")
            else:
                pass
        except:
            pass
