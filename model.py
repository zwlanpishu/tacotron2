from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths


class LocationLayer(nn.Module):
    def __init__(
        self, attention_n_filters, attention_kernel_size, attention_dim
    ):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh"
        )

        self.query_aux_layer = LinearNorm(
            attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh"
        )

        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False, w_init_gain="tanh"
        )

        self.memory_aux_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False, w_init_gain="tanh"
        )

        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.v_aux = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim,
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
        self, query, processed_memory, attention_weights_cat
    ):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(
            attention_weights_cat
        )
        energies = self.v(
            torch.tanh(
                processed_query
                + processed_attention_weights
                + processed_memory
            )
        )

        energies = energies.squeeze(-1)
        return energies, processed_query

    def get_align_aux_energies(self, query, processed_memory):
        processed_query_aux = self.query_aux_layer(query.unsqueeze(1))
        energies = self.v_aux(
            torch.tanh(processed_query_aux + processed_memory)
        )
        energies = energies.squeeze(-1)
        return energies

    def forward(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask,
        memory_aux,
        processed_aux,
        mask_aux,
    ):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        option: another decoder information
        """
        alignment, processed_query = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        if memory_aux is not None:
            alignment_aux = self.get_align_aux_energies(
                attention_hidden_state, processed_aux
            )
            if mask_aux is not None:
                alignment_aux.data.masked_fill_(
                    mask_aux, self.score_mask_value
                )
            attention_aux = F.softmax(alignment_aux, dim=1)
            context_aux = torch.bmm(attention_aux.unsqueeze(1), memory_aux)
            context_aux = context_aux.squeeze(1)
            attention_context += context_aux
        else:
            attention_aux = None

        return (
            attention_context,
            attention_weights,
            processed_query,
            attention_aux,
        )


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    hparams.n_mel_channels,
                    hparams.postnet_embedding_dim,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=int((hparams.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(hparams.postnet_embedding_dim),
            )
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        hparams.postnet_embedding_dim,
                        hparams.postnet_embedding_dim,
                        kernel_size=hparams.postnet_kernel_size,
                        stride=1,
                        padding=int((hparams.postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    hparams.postnet_embedding_dim,
                    hparams.n_mel_channels,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=int((hparams.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(hparams.n_mel_channels),
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(
                torch.tanh(self.convolutions[i](x)), 0.5, self.training
            )
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    hparams.encoder_embedding_dim,
                    hparams.encoder_embedding_dim,
                    kernel_size=hparams.encoder_kernel_size,
                    stride=1,
                    padding=int((hparams.encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(hparams.encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            hparams.encoder_embedding_dim,
            int(hparams.encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True
        )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams, direction):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold

        assert direction == "left" or direction == "right"
        if direction == "left":
            self.n_frames_per_step = hparams.n_frames_left
        else:
            self.n_frames_per_step = hparams.n_frames_right

        self.prenet = Prenet(
            hparams.n_mel_channels * self.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim],
        )

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim, hparams.attention_rnn_dim,
        )

        self.attention_layer = Attention(
            hparams.attention_rnn_dim,
            hparams.encoder_embedding_dim,
            hparams.attention_dim,
            hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size,
        )

        self.decoder_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim,
        )

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * self.n_frames_per_step,
        )

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            self.n_frames_per_step,
            bias=True,
            w_init_gain="sigmoid",
        )

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(
            memory.data.new(
                B, self.n_mel_channels * self.n_frames_per_step
            ).zero_()
        )
        return decoder_input

    def initialize_decoder_states(self, memory, mask, option):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(
            memory.data.new(B, self.attention_rnn_dim).zero_()
        )
        self.attention_cell = Variable(
            memory.data.new(B, self.attention_rnn_dim).zero_()
        )

        self.decoder_hidden = Variable(
            memory.data.new(B, self.decoder_rnn_dim).zero_()
        )
        self.decoder_cell = Variable(
            memory.data.new(B, self.decoder_rnn_dim).zero_()
        )

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(
            memory.data.new(B, MAX_TIME).zero_()
        )
        self.attention_context = Variable(
            memory.data.new(B, self.encoder_embedding_dim).zero_()
        )

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

        if option is not None:
            self.memory_aux = option[0]
            MAX_AUX = option[0].size(1)
            if option[1] is not None:
                mask_origin = ~get_mask_from_lengths(option[1])
                B, T, _ = option[0].size()
                T_origin = mask_origin.size(1)
                mask_aux = mask_origin.new_ones(B, T)
                mask_aux[:, :T_origin] = mask_origin
                self.mask_aux = mask_aux
            else:
                self.mask_aux = None
            self.processed_aux = self.attention_layer.memory_aux_layer(
                option[0]
            )
            self.attention_aux = Variable(
                option[0].data.new(B, MAX_AUX).zero_()
            )
        else:
            self.memory_aux = None
            self.mask_aux = None
            self.processed_aux = None
            self.attention_aux = None

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.reshape(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(
        self,
        mel_outputs,
        gate_outputs,
        alignments,
        att_regulars,
        alignments_aux,
    ):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.reshape(
            mel_outputs.size(0), -1, self.n_mel_channels
        )
        gate_outputs = gate_outputs.reshape(gate_outputs.size(0), 1, -1)
        gate_outputs = gate_outputs.squeeze(1)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        att_regulars = torch.stack(att_regulars).transpose(0, 1).contiguous()

        if len(alignments_aux) != 0:
            alignments_aux = torch.stack(alignments_aux).transpose(0, 1)
        else:
            alignments_aux = None

        return (
            mel_outputs,
            gate_outputs,
            alignments,
            att_regulars,
            alignments_aux,
        )

    def decode(self, decoder_input, option=None):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        option: information of another decoder outputs, (N, T, C)

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """

        prenet_output = self.prenet(decoder_input)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            prenet_output, (self.attention_hidden, self.attention_cell)
        )

        attention_weights_cat = torch.cat(
            (
                self.attention_weights.unsqueeze(1),
                self.attention_weights_cum.unsqueeze(1),
            ),
            dim=1,
        )

        (
            self.attention_context,
            self.attention_weights,
            processed_query,
            self.attention_aux,
        ) = self.attention_layer(
            self.attention_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask,
            self.memory_aux,
            self.processed_aux,
            self.mask_aux,
        )

        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat((prenet_output, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        )
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context
        )

        att_regular = torch.cat(
            (processed_query.squeeze(1), self.attention_context), dim=1
        )

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return (
            decoder_output,
            gate_prediction,
            self.attention_weights,
            att_regular,
            self.attention_aux,
        )

    def forward(
        self, memory, decoder_inputs, memory_lengths, teacher, option=None
    ):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.
        teacher: determine to use teacher forcing mode or free run mode.
        option: receive the auxiliary information of another decoder.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        att_regular: the concatenation of query and context at each step
        alignments_left: sequence of attention weights for option
        """

        decoder_input = self.get_go_frame(memory)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths), option=option
        )

        mel_outputs, gate_outputs, alignments, att_regulars = [], [], [], []
        alignments_aux = []

        while len(mel_outputs) < decoder_inputs.size(0):
            (
                mel_output,
                gate_output,
                attention_weights,
                att_regular,
                attention_aux,
            ) = self.decode(decoder_input, option)
            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [attention_weights]
            att_regulars += [att_regular]

            if option is not None:
                alignments_aux += [attention_aux]

            if teacher is True:
                decoder_input = decoder_inputs[len(mel_outputs) - 1]
            else:
                mel_temp = mel_outputs[-1].detach()
                decoder_input = mel_temp

        (
            mel_outputs,
            gate_outputs,
            alignments,
            att_regulars,
            alignments_aux,
        ) = self.parse_decoder_outputs(
            mel_outputs,
            gate_outputs,
            alignments,
            att_regulars,
            alignments_aux,
        )

        return (
            mel_outputs,
            gate_outputs,
            alignments,
            att_regulars,
            alignments_aux,
        )

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments, att_regulars = [], [], [], []
        while True:
            mel_output, gate_output, alignment, att_regular = self.decode(
                decoder_input
            )

            mel_outputs += [mel_output]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [alignment]
            att_regulars += [att_regular]

            eos = F.sigmoid(gate_output).detach().cpu().numpy()

            if True in (eos >= self.gate_threshold):
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        # add for delete noise, the last frame is just a token
        mel_outputs.pop()
        gate_outputs.pop()
        alignments.pop()
        att_regulars.pop()

        (
            mel_outputs,
            gate_outputs,
            alignments,
            att_regulars,
        ) = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, att_regulars
        )

        return mel_outputs, gate_outputs, alignments, att_regulars


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim, max_norm=1.0
        )
        self.encoder = Encoder(hparams)
        self.decoder_l = Decoder(hparams, "left")
        self.linear_l2r = LinearNorm(
            hparams.n_mel_channels, hparams.encoder_embedding_dim, bias=False
        )
        self.decoder_r = Decoder(hparams, "right")
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
        ) = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded["left"] = to_gpu(mel_padded["left"]).float()
        mel_padded["right"] = to_gpu(mel_padded["right"]).float()
        gate_padded["left"] = to_gpu(gate_padded["left"]).float()
        gate_padded["right"] = to_gpu(gate_padded["right"]).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded),
        )

    def parse_output_left(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths + 1)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            B, C, T = outputs[0].size()
            T_m = mask.size(2)
            mask_padded = mask.new_ones(B, C, T)
            mask_padded[:, :, :T_m] = mask

            outputs[0].data.masked_fill_(mask_padded, 0.0)
            outputs[1].data.masked_fill_(mask_padded[:, 0, :], 1e3)
        return outputs

    def parse_output_right(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths + 1)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            B, C, T = outputs[0].size()
            T_m = mask.size(2)
            mask_padded = mask.new_ones(B, C, T)
            mask_padded[:, :, :T_m] = mask

            outputs[0].data.masked_fill_(mask_padded, 0.0)
            outputs[1].data.masked_fill_(mask_padded, 0.0)
            outputs[2].data.masked_fill_(mask_padded[:, 0, :], 1e3)
        return outputs

    def forward(self, inputs, teacher=True):
        text_inputs, text_lengths, mels, _, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        # the left forward decoding
        mel_outs_l, gate_outs_l, aligns_l, att_regulars_l, _ = self.decoder_l(
            encoder_outputs,
            mels["left"],
            memory_lengths=text_lengths,
            teacher=teacher,
        )

        # the right forward decoding
        mel_outs_l_temp = mel_outs_l.transpose(1, 2)
        mel_outs_l_temp = self.linear_l2r(mel_outs_l_temp)
        auxiliary_info = [mel_outs_l_temp] + [output_lengths]
        (
            mel_outs_r,
            gate_outs_r,
            aligns_r,
            att_regulars_r,
            aligns_aux,
        ) = self.decoder_r(
            encoder_outputs,
            mels["right"],
            memory_lengths=text_lengths,
            teacher=teacher,
            option=auxiliary_info,
        )

        # the post process
        mel_outputs_postnet = self.postnet(mel_outs_r)
        mel_outputs_postnet = mel_outs_r + mel_outputs_postnet

        outputs_left = self.parse_output_left(
            [mel_outs_l, gate_outs_l, aligns_l, att_regulars_l],
            output_lengths,
        )
        outputs_right = self.parse_output_right(
            [
                mel_outs_r,
                mel_outputs_postnet,
                gate_outs_r,
                aligns_r,
                att_regulars_r,
                aligns_aux,
            ],
            output_lengths,
        )
        return outputs_left, outputs_right

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        (
            mel_outputs,
            gate_outputs,
            alignments,
            att_regulars,
        ) = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [
                mel_outputs,
                mel_outputs_postnet,
                gate_outputs,
                alignments,
                att_regulars,
            ]
        )

        return outputs
