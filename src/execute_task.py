import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)
client_socket.connect(server_address)
try:
    message = 'detection'
    print('Sending: ', message)
    client_socket.sendall(message.encode())
finally:
    client_socket.close()