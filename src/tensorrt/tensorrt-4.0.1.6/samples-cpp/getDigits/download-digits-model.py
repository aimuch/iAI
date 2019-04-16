#!/usr/bin/env python

import sys
import os.path
import argparse
import requests
import json

class ModelDownloader(object):
    """
    Downloads a DIGITS model
    """

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port

    def get_url(self, url):
        try:
            r = requests.get(url)
            r.raise_for_status()
            return r.content
        except requests.exceptions.ConnectionError as e:
            print 'Failed to connect to server at %s:%s' % (self.hostname, self.port)
        except requests.exceptions.RequestException as e:
            print 'Error loading "%s"' % url
            print '\t', e.message
        sys.exit(1)

    def get_job_id(self):
        """
        Present the user with a list of models on the server and return the id of the selection
        """
        url = 'http://%s:%s/index.json' % (self.hostname, self.port)
        models = json.loads(self.get_url(url))['models']
        if not len(models):
            raise Exception('No models exist on this server!')
        fmt = '[%3s] %-20s %-10s %-20s'
        print fmt % ('Num', 'Job ID', 'Status', 'Name')
        print '-' * len(fmt % ('a', 'a', 'a', 'a'))
        for i, model in enumerate(models):
            print fmt % (i+1, model['id'], model['status'], model['name'])

        selected = None
        while selected is None:
            print 'Select a job'
            x = raw_input('>>> ')
            try:
                x = int(x)-1
                if 0 <= x < len(models):
                    selected = x
                else:
                    print 'Out of range'
            except ValueError as e:
                print e
        print

        return models[selected]['id']

    def get_snapshot_epoch(self, job_id):
        """
        Present the user with a list of snapshots and return the epoch of the selection
        """
        url = 'http://%s:%s/models/%s.json' % (self.hostname, self.port, job_id)
        snapshots = json.loads(self.get_url(url))['snapshots']
        if not len(snapshots):
            raise Exception('No snapshots exist for this job!')
        fmt = '[%3s] %-10s'
        print fmt % ('Num', 'Epoch')
        print '-' * len(fmt % ('a', 'a'))
        for i, epoch in enumerate(snapshots):
            print fmt % (i+1, epoch)

        selected = None
        default = len(snapshots)
        while selected is None:
            print 'Select a snapshot (leave blank for default=%s)' % default
            x = raw_input('>>> ')
            if not x.strip():
                selected = default - 1
            else:
                try:
                    x = int(x)-1
                    if 0 <= x < len(snapshots):
                        selected = x
                    else:
                        print 'Out of range'
                except ValueError as e:
                    print e
        print

        epoch = snapshots[selected]
        return epoch

    def download_model(self, output_file, job_id, snapshot_epoch):
        """
        Download a tarfile of a specific snapshot
        """
        extension = '.'.join([''] + os.path.basename(output_file).split('.')[1:])

        url = 'http://%s:%s/models/%s/download%s?epoch=%s' % (self.hostname, self.port, job_id, extension, snapshot_epoch)
        content = self.get_url(url)

        print 'Saving to %s' % output_file
        with open(output_file, 'wb') as outfile:
            outfile.write(content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download a DIGITS model tarball')

    ### Positional arguments

    parser.add_argument('output_file',
            help='output file (should end with .zip, .tar, .tar.gz or .tar.bz2)')

    ### Optional arguments

    parser.add_argument('-n', '--hostname',
            default='127.0.0.1',
            help='hostname for the DIGITS server [default=127.0.0.1]')

    parser.add_argument('-p', '--port',
            type=int,
            default=80,
            help='port for the DIGITS server [default=80]')

    args = vars(parser.parse_args())

    downloader = ModelDownloader(args['hostname'], args['port'])
    job_id = downloader.get_job_id()
    snapshot_epoch = downloader.get_snapshot_epoch(job_id)
    downloader.download_model(args['output_file'], job_id, snapshot_epoch)

    print 'Done.'

