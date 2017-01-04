#!/usr/bin/python
#!/bin/env python

import argparse
import cmath
from source_alphabet import source_alphabet
from gnuradio import channels, digital, gr, blocks
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft, cPickle, gzip
import logging
import random
import sys


# define utility function(s)
def create_unit_vector(complex_num, iq_swap=False):
    magnitude = math.sqrt(np.real(complex_num) ** 2 + np.imag(complex_num) ** 2)
    if iq_swap == False:
        return (np.real(complex_num) / magnitude, np.imag(complex_num) / magnitude)
    else:
        return (np.imag(complex_num) / magnitude, np.real(complex_num) / magnitude)


# define constants for data parsing
DYNAMIC_CHANNEL_SETTLING_TIME_IN_SAMPLES = 1000

# define constants/defaults for the command-line parser
DESCRIPTION = "Generate a series of IQ vectors that represent GMSK signal(s) in a dynamic" + \
              " channel model across a range of SNRs."

PASSTHROUGH_CHANNEL_MODEL_STR = "passthrough"       # passthrough; no channel effects
DYNAMIC_CHANNEL_MODEL_STR = "dynamic"               # dynamic channel; no GMSK interference
INTERFERENCE_CHANNEL_MODEL_STR = "interference"     # dynamic channel + GMSK interference

ALLOWED_CHANNEL_MODELS = [PASSTHROUGH_CHANNEL_MODEL_STR, DYNAMIC_CHANNEL_MODEL_STR, INTERFERENCE_CHANNEL_MODEL_STR]
DEFAULT_CHANNEL_MODEL_STR = DYNAMIC_CHANNEL_MODEL_STR

DEFAULT_OUTPUT_FILE = "gmsk_dataset.dat"

DEFAULT_SAMPLES_PER_SYMBOL = 8
DEFAULT_SYMBOLS_PER_BURST = 156
DEFAULT_VECTOR_LENGTH_IN_SYMBOLS = DEFAULT_SYMBOLS_PER_BURST * 3
DEFAULT_KNOWN_SYMBOLS_PER_BURST = 26
DEFAULT_NUM_IQ_VECTORS = 4096

DEFAULT_MINIMUM_SNR_VALUE = 0
DEFAULT_MAXIMUM_SNR_VALUE = 20
SNR_VALUE_STEP_SIZE = 2


# create the command-line argument parser
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--channel_model', default=DEFAULT_CHANNEL_MODEL_STR, \
                    help="Channel model: <%s|%s|%s>; defaults to %s" % \
                    (PASSTHROUGH_CHANNEL_MODEL_STR, DYNAMIC_CHANNEL_MODEL_STR, INTERFERENCE_CHANNEL_MODEL_STR,
                     DEFAULT_CHANNEL_MODEL_STR))
parser.add_argument('--known_symbols_per_burst', default=DEFAULT_KNOWN_SYMBOLS_PER_BURST, type=int, \
                    help="Number of a priori known symbols per 'burst'; defaults to %u" % (DEFAULT_KNOWN_SYMBOLS_PER_BURST))
parser.add_argument('--maximum_snr', default=DEFAULT_MAXIMUM_SNR_VALUE, type=int, \
                    help="Maximum SNR for dataset vectors; defaults to %u" % (DEFAULT_MAXIMUM_SNR_VALUE))
parser.add_argument('--minimum_snr', default=DEFAULT_MINIMUM_SNR_VALUE, type=int, \
                    help="Minimum SNR for dataset vectors; defaults to %u" % (DEFAULT_MINIMUM_SNR_VALUE))
parser.add_argument('--num_vectors', default=DEFAULT_NUM_IQ_VECTORS, type=int, \
                    help="Number of vectors to generate at each SNR; defaults to %u" % DEFAULT_NUM_IQ_VECTORS)
parser.add_argument('--output_file', default=DEFAULT_OUTPUT_FILE, \
                    help="Output file name; defaults to %s" % (DEFAULT_OUTPUT_FILE))
parser.add_argument('--samples_per_symbol', default=DEFAULT_SAMPLES_PER_SYMBOL, type=int, \
                    help="Number of quadrature samples per GMSK symbol; defaults to %u" % (DEFAULT_SAMPLES_PER_SYMBOL))
parser.add_argument('--symbols_per_burst', default=DEFAULT_SYMBOLS_PER_BURST, type=int, \
                    help="Number of GMSK symbols that form a single 'burst'; defaults to %u" % (DEFAULT_SYMBOLS_PER_BURST))
parser.add_argument('--vector_length', default=DEFAULT_VECTOR_LENGTH_IN_SYMBOLS, type=int, \
                    help="Vector length in GMSK symbols; defaults to %u" % DEFAULT_VECTOR_LENGTH_IN_SYMBOLS)


# parse the command-line arguments
args = parser.parse_args()


# setup the logger
logging.basicConfig(level=logging.INFO, \
                    format='%(asctime)s [GMSK_DATA] [%(levelname)s] %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p')


# perform custom command-line parsing validation
if args.channel_model not in ALLOWED_CHANNEL_MODELS:
    logging.error("Channel model %s is not supported" % args.channel_model)
    sys.exit(-1)

if args.known_symbols_per_burst > args.symbols_per_burst:
    logging.error("Number of known symbols per burst (%u) cannot exceed symbols per burst (%u)!!!" % \
        (args.known_symbols_per_burst, args.symbols_per_burst))
    sys.exit(-1)

if args.maximum_snr < args.minimum_snr:
    logging.error("Maximum SNR must be greater than minimum SNR!!!")
    sys.exit(-1)

if args.samples_per_symbol not in [2, 4, 8]:
    logging.error("Invalid number of samples per symbol. Choose from [2,4,8]")
    sys.exit(-1)

if args.vector_length < args.symbols_per_burst:
    logging.error("Number of symbols per vector cannot be less than the number of symbols per burst!!!")
    sys.exit(-1)

if (args.symbols_per_burst - args.known_symbols_per_burst) % 2 != 0:
    logging.error("Values violate the assumption that (symbols_per_burst - known_symbols_per_burst) is evenly divisible by 2")
    sys.exit(-1)

if (args.vector_length % args.symbols_per_burst != 0):
    logging.error("Values violate the assumption that there are an integer number of bursts in the vector!!!")
    sys.exit(-1)


# calculate the vector length in terms of quadrature samples; the user provides
#  a vector length in GMSK symbols, so we need to calculate the corresponding
#  quadrature vector length
IQ_VECTOR_LENGTH = args.samples_per_symbol * args.vector_length
logging.debug("Calculated quadrature vector length of %u samples" % \
    (IQ_VECTOR_LENGTH))


# known symbols are found in the middle of 'bursts'. calculate the corresponding
#  sample indices for these known values.
known_sample_indicies = []

# calculate the known sample indicies for a single burst
samples_per_burst = args.symbols_per_burst * args.samples_per_symbol
unknown_samples_per_burst = (args.symbols_per_burst - args.known_symbols_per_burst) * args.samples_per_symbol
known_samples_per_burst = args.known_symbols_per_burst * args.samples_per_symbol

# now extend a single burst to the number of bursts that appear in
#  each of the data vectors.
for burst in range(int(math.ceil(IQ_VECTOR_LENGTH / (samples_per_burst)))):
    for sample in range(samples_per_burst):

        # calculate the next sample index
        cur_sample_idx = burst * samples_per_burst + sample

        # save if this sample corresponds to one of the 'known' symbols
        if sample >= (unknown_samples_per_burst / 2) and \
           sample < ((unknown_samples_per_burst / 2) + known_samples_per_burst):
            known_sample_indicies.append(cur_sample_idx)


# in addition to the known samples/symbols, the network is expected to predict
#  symbol values that correspond to certain points in the data vector.
#  specifically, it should predict the symbol values for the 'middle' burst in
#  the data vector.
num_bursts_in_vector = int(math.ceil(IQ_VECTOR_LENGTH / (samples_per_burst)))
middle_burst = num_bursts_in_vector / 2
predicted_symbol_offset = middle_burst * args.symbols_per_burst

logging.info("%u bursts of %u symbols in data vector; predicted symbols start at symbol %u" % \
    (num_bursts_in_vector, args.symbols_per_burst, predicted_symbol_offset))


# create the output dataset, which is a dictionary with two top-level elements:
#  'data' and 'metadata'. the 'data' element is a list of dictionaries where
#  dictionary has the format:
#
# {'modulation':'gmsk', 'snr':<snr>, 'channel_model':<'passthrough'|'dynamic'|'interference'>,
#  'data': np.array(5, IQ_VECTOR_LENGTH),
#  'prediction': np.array(1, args.symbols_per_burst)}
#
# The 'data' np.array contains the following contents:
#  index    description
#    0        measured quadrature sample (converted to unit vector); real component
#    1        measured quadrature sample (converted to unit vector); imaginary component
#    2        original quadrature sample (converted to unit vector); real component
#    3        original quadrature sample (converted to unit vector); imaginary component
#    4        boolean indicating whether the original value is valid
#
# The 'prediction' np.array contains the following contents:
#  index    description
#    0        boolean indicating the symbol values for the 'middle' burst from 'data'
generated_dataset = {'data':[], 'metadata':{}}

# define constants for the 'data' array indicies
SAMPLED_IQ_I_IDX = 0
SAMPLED_IQ_Q_IDX = 1
REFERENCE_IQ_I_IDX = 2
REFERENCE_IQ_Q_IDX = 3
REFERENCE_IQ_VALID_IDX = 4


# add 'metadata' information about the generated dataset
generated_dataset['metadata']['samples_per_symbol'] = args.samples_per_symbol
generated_dataset['metadata']['symbols_per_burst']  = args.symbols_per_burst

logging.info("command-line options:")
logging.info("  samples_per_symbol=%u" % (args.samples_per_symbol))
logging.info("  symbols_per_burst =%u" % (args.symbols_per_burst))

# define the SNR range to use for creating the dataset
snr_vals = range(args.minimum_snr, args.maximum_snr + 1, SNR_VALUE_STEP_SIZE)
logging.info("Generating vectors over SNR range [%u,%u]" % (args.minimum_snr, args.maximum_snr))


# the source data is common across all SNR values and dynamic channel effects; start
#  by using GNU Radio to generate the reference data.
num_symbols_to_generate = int(1e5)


# now apply various dynamic channel effects; repeat at each SNR value
for snr in snr_vals:

    # create a new dataset; this will be filled in for each new snr value
    snr_generated_dataset = []

    logging.info("Generating data for SNR %d" % (snr))

    # start generating the data; note that we need args.num_vectors
    insufficient_snr_vectors = True
    while insufficient_snr_vectors:

        # create the source block. note that the num_symbols_to_generate is
        #  divided by 8 here because the source_alphabet block is NOT unpacking
        #  the char stream it receives. therefore, there are eight symbols in
        #  each element that it sends to the gmsk_mod block. these are unpacked
        #  in the gmsk_mod block and processed individually.
        snr_source_block = source_alphabet('discrete', num_symbols_to_generate / 8, False, True)

        # create a conversion block and a sink for raw symbol data; this allows
        #  us to tap off the symbol data before it is passed to the modulator.
        symbol_conversion_block = blocks.packed_to_unpacked_bb(1, gr.GR_LSB_FIRST)
        symbol_vals_sink = blocks.vector_sink_b()

        # create the gmsk modulation block
        snr_gmsk_mod_block = digital.gmsk_mod(args.samples_per_symbol)

        # create a sink block for the cleanly modulated data
        modulated_sink_block = blocks.vector_sink_c()

        # define the channel model parameters
        sample_rate_offset_std_dev = 0.01 # 0.01
        sample_rate_offset_max_dev = 0    # 50

        carrier_freq_offset_std_dev = 0.01
        carrier_freq_offset_max_dev = 0 # 0.5e3

        doppler_frequency = 0

        delays = [0.0, 0.9, 1.7]
        mags = [1, 0.8, 0.3]
        n_filter_taps = 8
        noise_amp = 10**(-snr/10.0)

        # create the GNU Radio channel model block
        channel_model_block = channels.dynamic_channel_model( 200e3,
                                                              sample_rate_offset_std_dev,
                                                              sample_rate_offset_max_dev,
                                                              carrier_freq_offset_std_dev,
                                                              carrier_freq_offset_max_dev,
                                                              8,
                                                              doppler_frequency,
                                                              True, 8, delays, mags, \
                                                              n_filter_taps, noise_amp, 0x1337 )

        # create a sink block for the signal as transmitted through the channel
        channel_sink_block = blocks.vector_sink_c()

        # create the top-level GNU Radio block
        top_block = gr.top_block()

        # connect the various blocks

        # start by connecting the source, symbol sink path, modulator, and
        #  modulator sink blocks
        top_block.connect(snr_source_block, snr_gmsk_mod_block)
        top_block.connect(snr_source_block, symbol_conversion_block, symbol_vals_sink)
        top_block.connect(snr_gmsk_mod_block, modulated_sink_block)

        # next, connect blocks based on the command-line settings
        if args.channel_model != PASSTHROUGH_CHANNEL_MODEL_STR:
            logging.debug("Using a dynamic channel model")

            # provides IQ data as transmitted through the channel
            top_block.connect(snr_gmsk_mod_block, channel_model_block, channel_sink_block)

        else:
            logging.debug("Using a pass-through channel model")

            # provides the modulated IQ data; repeat of the tap off into the
            #  modulated_sink_block.
            top_block.connect(snr_gmsk_mod_block, channel_sink_block)

        # run the GNU Radio 'assembly'
        top_block.run()


        ########################################
        # get the REFERNCE SYMBOL data
        ########################################

        original_symbol_vector = np.array(symbol_vals_sink.data(), dtype=np.int8)

        # in order to facilitate soft symbol predictions from a neural network, change
        #  all of the '0' values into '-1' values.
        for i in range(len(original_symbol_vector)):
            if original_symbol_vector[i] == 0:
                original_symbol_vector[i] = -1

        logging.info("Generated %u unmodulated symbols" %
            (len(original_symbol_vector)))


        ########################################
        # get the REFERENCE IQ data
        ########################################

        original_modulation_vector = np.array(modulated_sink_block.data(), dtype=np.complex64)
        logging.info("Generated %u ideally modulated IQ samples; using %u samples per symbol" %
            (len(original_modulation_vector), len(original_modulation_vector) / num_symbols_to_generate))


        ########################################
        # get the DYNAMIC IQ data
        ########################################

        dynamic_channel_output_vector = np.array(channel_sink_block.data(), dtype=np.complex64)
        logging.info("Generated new dynamic channel dataset (%u samples); starting to extract test vectors" % \
            (len(dynamic_channel_output_vector)))

        # check for a minimum number of samples
        if len(dynamic_channel_output_vector) < (args.vector_length + DYNAMIC_CHANNEL_SETTLING_TIME_IN_SAMPLES):
            logging.error("Generated a data vector of %u samples when at least %u are needed; check inputs..." % \
                (len(dynamic_channel_output_vector), (args.vector_length + DYNAMIC_CHANNEL_SETTLING_TIME_IN_SAMPLES)))
            sys.exit(-1)

        # we want to sample from the generated data stream some random time after the channel
        #  model transients have settled. values chosen here are arbitrary.
        sample_index = random.randint(50, DYNAMIC_CHANNEL_SETTLING_TIME_IN_SAMPLES)

        # make sure that the sample index is on a symbol boundary
        if sample_index % args.samples_per_symbol != 0:
            sample_index += (args.samples_per_symbol - (sample_index % args.samples_per_symbol))

        # verify that the sample_index was adjusted correctly
        if sample_index % args.samples_per_symbol != 0:
            logging.error("Sample index is NOT on a symbol boundary!!!")
            sys.exit(-1)


        while (sample_index + IQ_VECTOR_LENGTH) < len(dynamic_channel_output_vector) and \
              (sample_index + IQ_VECTOR_LENGTH) < len(original_modulation_vector) and \
              insufficient_snr_vectors == True:

            # get the next sample vector
            sampled_vector = dynamic_channel_output_vector[sample_index:sample_index + IQ_VECTOR_LENGTH]
            logging.debug("Retrieved %u samples from [%u:%u]" % (len(sampled_vector), sample_index, sample_index + IQ_VECTOR_LENGTH))

            # get the corresponding symbol value vector
            vector_start_symbol = (sample_index / args.samples_per_symbol) + predicted_symbol_offset
            symbol_vals = original_symbol_vector[vector_start_symbol:vector_start_symbol + args.symbols_per_burst]
            logging.debug("Retrieved %u symbols from [%u,%u]" % (len(symbol_vals), vector_start_symbol, vector_start_symbol + args.symbols_per_burst))

            # get the corresponding ideally modulated vector
            original_vector = original_modulation_vector[sample_index:sample_index + IQ_VECTOR_LENGTH]
            logging.debug("Retrieved %u original samples from [%u,%u]" % (len(original_vector), sample_index, sample_index + IQ_VECTOR_LENGTH))

            # create a new vector data structure
            cur_output = {'modulation':'gmsk', 'snr':snr, 'channel_model':args.channel_model, \
                          'data':np.zeros([5, IQ_VECTOR_LENGTH], dtype=np.float32),
                          'prediction':np.zeros([1 * args.symbols_per_burst], dtype=np.int8)}

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                # convert each IQ sample to a unit vector
                for i in range(len(sampled_vector)):

                    x1, y1, x2, y2, x3, y3 = [], [], [], [], [], []
                    for j in range(args.samples_per_symbol * 80):
                        # get the original vector
                        real, imag = create_unit_vector(original_vector[j])
                        x1.append(real)
                        y1.append(imag)

                        # get the version that has been converted to a unit vector
                        if args.channel_model != PASSTHROUGH_CHANNEL_MODEL_STR:
                            real, imag = create_unit_vector(sampled_vector[j], True)
                            x3.append(real * -1)
                            y3.append(imag)
                        else:
                            real, imag = create_unit_vector(sampled_vector[j], False)
                            x3.append(real)
                            y3.append(imag)

                    plt.figure(1)
                    plt.subplot(411)
                    plt.plot(x1)
                    plt.grid(True)
                    plt.subplot(412)
                    plt.plot(y1)
                    plt.grid(True)

                    plt.subplot(413)
                    plt.plot(x3)
                    plt.grid(True)
                    plt.subplot(414)
                    plt.plot(y3)
                    plt.grid(True)

                    plt.show()
                    sys.exit(0)

                # convert the original and sampled data to unit vectors; this
                #  is providing a data regularization function. Note that based
                #  on empirical observation it is necessary to do an IQ swap
                #  on the sampled data.
                original_i, original_q = create_unit_vector(original_vector[i])
                sampled_i, sampled_q = create_unit_vector(sampled_vector[i], True)

                # save the sampled data into the 'data' vector. Note that based
                #  on empirical observation it is necessary to reverse the sign
                #  on the I data so that the saved data matches the original IQ
                cur_output['data'][SAMPLED_IQ_I_IDX][i] = sampled_i * -1
                cur_output['data'][SAMPLED_IQ_Q_IDX][i] = sampled_q

                # is this one of the samples that has an a priori known value? if so, then
                #  save the original value into the 'data' vector too.
                if i in known_sample_indicies:

                    cur_output['data'][REFERENCE_IQ_I_IDX][i] = original_i
                    cur_output['data'][REFERENCE_IQ_Q_IDX][i] = original_q

                    # set a flag indicating that this value is meaningful
                    cur_output['data'][REFERENCE_IQ_VALID_IDX][i] = 1.0


            # all 'data' values have been saved. now save the 'predicted' values
            for i in range(args.symbols_per_burst):
                cur_output['prediction'][i] = symbol_vals[i]


            # add the current output to the dataset for this SNR level
            snr_generated_dataset.append(cur_output)

            # update the next sample index. goal here is to make sure that we only take a few
            #  sample vectors from each independent sample (i.e. force the model to regenerate
            #  a new dynamic channel often)
            sample_index += random.randint(IQ_VECTOR_LENGTH, round(len(dynamic_channel_output_vector) * 0.05))

            # see if we have enough data for this SNR level
            if len(snr_generated_dataset) == args.num_vectors:
                insufficient_snr_vectors = False

        logging.info("Generated %u of %u vectors for SNR %u" % \
            (len(snr_generated_dataset), args.num_vectors, snr))

    # copy data from the SNR list into the master list
    for data in snr_generated_dataset:
        generated_dataset['data'].append(data)

# shuffle the dataset
random.shuffle(generated_dataset['data'])

logging.info("Dataset generation is complete; writing to disk (%s)..." % (args.output_file))
cPickle.dump( generated_dataset, file(args.output_file, "wb" ) )
