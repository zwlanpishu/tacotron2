################################
# Experiment Parameters        #
################################
epochs = 2000
iters_per_checkpoint = 1000
seed = 1234
cudnn_enabled = True
cudnn_benchmark = False
trans_type = "phn"

################################
# Data Parameters             #
################################
load_mel_from_disk = True
data_files = "filelists/data.csv"
training_files = "filelists/train_set.csv"
validation_files = "filelists/dev_set.csv"
test_files = "filelists/test_set.csv"
custom_files = "filelists/custom_set.csv"
dump = "/home/server/disk1/DATA/LJS/LJSpeech-1.1/wavs"

################################
# Audio Parameters             #
################################

# if use tacotron 1's feature normalization
tacotron1_norm = False
preemphasis = 0.97
ref_level_db = 20.0
min_level_db = -100.0

sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

################################
# Model Parameters             #
################################
n_symbols = 35 if trans_type == "char" else 78
symbols_embedding_dim = 512

# Encoder parameters
encoder_kernel_size = 5
encoder_n_convolutions = 3
encoder_embedding_dim = 512

# Decoder parameters
n_frames_per_step = 2
decoder_rnn_dim = 1024
prenet_dim = 256
max_decoder_steps = 1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1
infer_trim = 2

# Attention parameters
attention_rnn_dim = 1024
attention_dim = 128

# Location Layer parameters
attention_location_n_filters = 32
attention_location_kernel_size = 31

# Mel-post processing network parameters
postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

################################
# Optimization Hyperparameters #
################################
learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
batch_size = 32
accum_size = 1
mask_padding = True
