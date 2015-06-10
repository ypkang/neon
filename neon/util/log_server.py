import pickle
import logging
import logging.handlers
import SocketServer
import struct


class LogRecordStreamHandler(SocketServer.StreamRequestHandler):
    """Handler for a streaming logging request"""

    def handle(self):
        """Handle multiple requests each expected to be a 4-byte length
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally."""

        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                # didn't get enough bytes in the length specifier
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                # what is going on here?
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = pickle.loads(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather
        # than the one implied by the record

        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        # every record gets logged
        # because logger.handle called after logger level filtering
        # if you want to do filtering, do it at the client
        # then you won't waste bandwidth
        logger.handle(record)

class LogRecordSocketReceiver(SocketServer.ThreadingTCPServer):
    # tcp socket based receiver for testing

    allow_reuse_address = 1

    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        SocketServer.ThreadingTCPServer.__init__(self, (host,port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()],
                                        [],[], self.timeout)
            if rd:
                self.handle_request()
                abort = self.abort

def main():
    logging.basicConfig(
        # created gives the time from server start in millis
        format='%(created)5d %(name)-15s %(levelname)-8s %(message)s')
    tcpserver = LogRecordSocketReceiver()
    print('Booting TCP Server')
    tcpserver.serve_until_stopped()

if __name__ == '__main__':
    main()
