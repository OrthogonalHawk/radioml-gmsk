#!/usr/bin/python
#!/bin/env python

# Modified version of source_alphabet.py obtained from:
#  https://github.com/radioML/dataset
#
# Removed the 'continuous' mode as it was unnecessary
#  for the GMSK investigation and relied on several out
#  of tree GNU Radio Blocks that were annoying to install.
#
# Also added another constructor parameter. The GNU Radio
#  gmsk_mod block already contains functionality that mimics
#  the blocks.packed_to_unpacked_bb behavior. to avoid a
#  double unpacking situation the unpacking here can be
#  bypassed.

from gnuradio import gr, blocks
import numpy as np
import os
import shutil
from subprocess import call
import sys

class source_alphabet(gr.hier_block2):

    MASTER_SOURCE_FILE = "source_material/gutenberg_shakespeare.txt"
    SPLIT_FILE_DIR = "/tmp/source_alphabet/"
    SOURCE_FILES = []
    CUR_SOURCE_FILE_IDX = 0
    SOURCE_SPLIT = False

    def __init__(self, dtype="discrete", limit=10000, unpack_bytes=False, randomize=False):

        if self.SOURCE_SPLIT == False:
            if os.path.exists(source_alphabet.SPLIT_FILE_DIR):
                shutil.rmtree(source_alphabet.SPLIT_FILE_DIR)

            # make the destination directory
            os.makedirs(source_alphabet.SPLIT_FILE_DIR)

            # split the source file into smaller chunks
            call("split --bytes 500K %s %ssource_piece_" % (source_alphabet.MASTER_SOURCE_FILE, source_alphabet.SPLIT_FILE_DIR), shell=True)

            # get a list of the resulting files
            source_alphabet.SOURCE_FILES = os.listdir(source_alphabet.SPLIT_FILE_DIR)

            # this only needs to be done once; set the flag
            source_alphabet.SOURCE_SPLIT = True

        if(dtype == "discrete"):
            gr.hier_block2.__init__(self, "source_alphabet",
                gr.io_signature(0,0,0),
                gr.io_signature(1,1,gr.sizeof_char))

            # select either the master source file or one of the split files
            if randomize:
                file_to_use = "%s%s" % (source_alphabet.SPLIT_FILE_DIR, source_alphabet.SOURCE_FILES[source_alphabet.CUR_SOURCE_FILE_IDX])
                print "Using %s" % (file_to_use)
                self.src = blocks.file_source(gr.sizeof_char, "%s" % (file_to_use))

                # on each use update the idx, but make sure it does not
                #  increment too far
                source_alphabet.CUR_SOURCE_FILE_IDX += 1
                source_alphabet.CUR_SOURCE_FILE_IDX %= len(source_alphabet.SOURCE_FILES)
            else:
                self.src = blocks.file_source(gr.sizeof_char, source_alphabet.MASTER_SOURCE_FILE)

            # optionally unpack the bytes
            if unpack_bytes:
                self.convert = blocks.packed_to_unpacked_bb(1, gr.GR_LSB_FIRST)
            else:
                self.convert = blocks.packed_to_unpacked_bb(8, gr.GR_LSB_FIRST)

            self.limit = blocks.head(gr.sizeof_char, limit)
            self.connect(self.src,self.convert)
            last = self.convert

        else:

            print "Unsupported type %s" % (dtype)
            sys.exit(-1)

        # connect head or not, and connect to output
        if(limit==None):
            self.connect(last, self)
        else:
            self.connect(last, self.limit, self)


if __name__ == "__main__":
    print "QA..."

    # Test discrete source
    tb = gr.top_block()
    src = source_alphabet("discrete", 1000)
    snk = blocks.vector_sink_b()
    tb.run()
